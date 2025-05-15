"""
Redis-based semantic cache for LLMs.

This module provides semantic caching of LLM prompts using Redis as the backend
storage and includes vector similarity search for semantic matching.
"""

import os
import json
import time
import hashlib
import logging
import numpy as np
import pickle
from typing import Dict, List, Any, Optional, Tuple, Union

import redis
from llama_cpp import Llama
from sentence_transformers import SentenceTransformer

# Set up logging
logger = logging.getLogger(__name__)


class RedisSemanticCache:
    """
    Redis-based semantic cache for LLMs.
    
    This cache stores LLM states indexed by both exact prompt matches and 
    semantic similarity, allowing for efficient retrieval of cached 
    results even when prompts are paraphrased or slightly modified.
    """
    
    def __init__(
        self,
        redis_host: str = "localhost",
        redis_port: int = 6379,
        redis_db: int = 0,
        redis_password: Optional[str] = None,
        model_name: str = "model",
        embedding_model: str = "all-MiniLM-L6-v2",
        vector_dim: int = 384,
        similarity_threshold: float = 0.85,
        ttl: int = 86400,  # 24 hours
        capacity_mb: int = 2048,  # 2GB default
        prefix: str = "llm:cache:",
        index_name: Optional[str] = None,
    ):
        """
        Initialize the Redis semantic cache.
        
        Args:
            redis_host: Redis server hostname
            redis_port: Redis server port
            redis_db: Redis database number
            redis_password: Redis password (if any)
            model_name: Name of the LLM model (used for namespacing)
            embedding_model: Name of the sentence transformer model to use for embeddings
            vector_dim: Dimension of embedding vectors
            similarity_threshold: Threshold for semantic similarity (0-1)
            ttl: Cache entry time-to-live in seconds
            capacity_mb: Max memory usage in MB
            prefix: Key prefix for Redis
            index_name: Redis search index name (default: derived from model_name)
        """
        self.redis_host = redis_host
        self.redis_port = redis_port
        self.redis_db = redis_db
        self.model_name = model_name
        self.prefix = f"{prefix}{model_name}:"
        self.vector_prefix = f"{prefix}{model_name}:vectors:"
        self.index_name = index_name or f"idx:{model_name}"
        self.vector_dim = vector_dim
        self.similarity_threshold = similarity_threshold
        self.ttl = ttl
        self.capacity_mb = capacity_mb
        
        # Connect to Redis
        self.redis = redis.Redis(
            host=redis_host,
            port=redis_port,
            db=redis_db,
            password=redis_password,
            decode_responses=True,  # Automatically decode responses to Python strings
        )
        
        # Also create a non-decoding client for binary data
        self.binary_redis = redis.Redis(
            host=redis_host,
            port=redis_port,
            db=redis_db,
            password=redis_password,
            decode_responses=False,  # Don't decode binary responses
        )
        
        # Set memory limit if capacity is provided
        if capacity_mb > 0:
            try:
                self.redis.config_set("maxmemory", f"{capacity_mb}mb")
                self.redis.config_set("maxmemory-policy", "allkeys-lru")
                logger.info(f"Redis memory limit set to {capacity_mb}MB with LRU eviction")
            except redis.exceptions.ResponseError as e:
                logger.warning(f"Could not set Redis memory limit: {e}")
                logger.warning("This may be due to running without proper permissions or a managed Redis instance")
        
        # Initialize sentence transformer for embeddings
        try:
            self.embedding_model = SentenceTransformer(embedding_model)
            logger.info(f"Loaded embedding model: {embedding_model}")
        except Exception as e:
            logger.error(f"Failed to load embedding model: {e}")
            self.embedding_model = None
            raise
        
        # Initialize the vector index
        self._initialize_vector_storage()
    
    def _initialize_vector_storage(self) -> None:
        """Initialize storage for vector embeddings"""
        logger.info(f"Initialized simple vector storage for semantic search")
    
    def compute_hash(self, tokens: List[int]) -> str:
        """
        Compute a hash from tokenized prompt for exact matching.
        
        Args:
            tokens: List of token IDs
            
        Returns:
            String hash representing the tokens
        """
        hash_obj = hashlib.sha256()
        for token in tokens:
            hash_obj.update(str(token).encode('utf-8'))
        return hash_obj.hexdigest()
    
    def compute_embedding(self, text: str) -> List[float]:
        """
        Compute embedding vector for a text string.
        
        Args:
            text: Text to embed
            
        Returns:
            Normalized embedding vector
        """
        if self.embedding_model is None:
            raise ValueError("Embedding model not initialized")
        
        # Get embedding and convert to list
        embedding = self.embedding_model.encode(text)
        
        # Ensure it's a standard float list for Redis
        return embedding.astype(np.float32).tolist()
    
    def get_key(self, hash_str: str) -> str:
        """Get Redis key for a hash"""
        return f"{self.prefix}{hash_str}"
    
    def get_vector_key(self, hash_str: str) -> str:
        """Get Redis key for a vector embedding"""
        return f"{self.vector_prefix}{hash_str}"
    
    def compute_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """
        Compute cosine similarity between two vectors.
        
        Args:
            vec1: First vector
            vec2: Second vector
            
        Returns:
            Cosine similarity score (0-1)
        """
        # Convert to numpy arrays
        np_vec1 = np.array(vec1)
        np_vec2 = np.array(vec2)
        
        # Compute cosine similarity
        dot_product = np.dot(np_vec1, np_vec2)
        norm1 = np.linalg.norm(np_vec1)
        norm2 = np.linalg.norm(np_vec2)
        
        # Avoid division by zero
        if norm1 == 0 or norm2 == 0:
            return 0.0
            
        return dot_product / (norm1 * norm2)
    
    def save_state(self, model: Llama, prompt: str, tokens: List[int]) -> None:
        """
        Save model state to cache.
        
        Args:
            model: Llama model instance
            prompt: Original prompt text
            tokens: Tokenized prompt
        """
        try:
            # Get hash from tokens for exact matches
            hash_str = self.compute_hash(tokens)
            key = self.get_key(hash_str)
            
            # Compute embedding for semantic matches
            embedding = self.compute_embedding(prompt)
            
            # Get model state
            state = model.save_state()
            
            # Serialize the state object using pickle
            state_bytes = pickle.dumps(state)
            
            # Create a record in Redis
            cache_data = {
                "prompt": prompt,
                "hash": hash_str,
                "timestamp": time.time(),
            }
            
            # Store metadata in a hash
            pipe = self.redis.pipeline()
            pipe.hset(key, mapping=cache_data)
            
            # Store embedding separately for semantic search
            vector_key = self.get_vector_key(hash_str)
            pipe.set(vector_key, json.dumps(embedding))
            
            # Store binary state data separately (using binary_redis client)
            state_key = f"{key}:state"
            self.binary_redis.set(state_key, state_bytes)
            
            # Set expiration if TTL is configured
            if self.ttl > 0:
                pipe.expire(key, self.ttl)
                pipe.expire(vector_key, self.ttl)
                self.binary_redis.expire(state_key, self.ttl)
            
            # Execute pipeline
            pipe.execute()
            
            logger.debug(f"Cached state for prompt hash {hash_str[:8]}...")
            
        except Exception as e:
            logger.error(f"Failed to save state to cache: {e}")
    
    def load_state(
        self, 
        model: Llama, 
        prompt: str = None, 
        tokens: List[int] = None,
        semantic_search: bool = True
    ) -> bool:
        """
        Load model state from cache.
        
        Args:
            model: Llama model instance
            prompt: Original prompt text (required for semantic search)
            tokens: Tokenized prompt (for exact match)
            semantic_search: Whether to perform semantic similarity search if exact match fails
        
        Returns:
            True if cache hit and state loaded successfully, False otherwise
        """
        if tokens is None and prompt is None:
            logger.error("Either prompt or tokens must be provided")
            return False
        
        try:
            # First try exact match if tokens are provided
            if tokens is not None:
                hash_str = self.compute_hash(tokens)
                key = self.get_key(hash_str)
                
                # Check if this exact prompt is in cache
                if self.redis.exists(key):
                    # Get binary state data
                    state_key = f"{key}:state"
                    state_bytes = self.binary_redis.get(state_key)
                    
                    if state_bytes:
                        # Deserialize the state object
                        state = pickle.loads(state_bytes)
                        
                        # Load state into model
                        model.load_state(state)
                        logger.debug(f"Loaded exact cache match for prompt hash {hash_str[:8]}...")
                        return True
            
            # If exact match fails and semantic search is enabled, try semantic search
            if semantic_search and prompt is not None:
                # Compute embedding for query
                query_embedding = self.compute_embedding(prompt)
                
                # Get all vector keys
                vector_keys = self.redis.keys(f"{self.vector_prefix}*")
                
                if not vector_keys:
                    return False
                
                # Find the best semantic match
                best_match = None
                best_similarity = 0
                
                # Get all vectors in batches to avoid large memory usage
                for key in vector_keys:
                    # Get the embedding
                    embedding_json = self.redis.get(key)
                    
                    if not embedding_json:
                        continue
                    
                    try:
                        embedding = json.loads(embedding_json)
                        
                        # Compute similarity
                        similarity = self.compute_similarity(query_embedding, embedding)
                        
                        # Update best match if this is better
                        if similarity > best_similarity:
                            best_similarity = similarity
                            # Extract hash from key
                            best_match = key.replace(self.vector_prefix, "")
                    except Exception as e:
                        logger.warning(f"Error processing vector {key}: {e}")
                
                # If we found a good enough match, load it
                if best_match and best_similarity >= self.similarity_threshold:
                    # Get the state from the cache
                    key = self.get_key(best_match)
                    state_key = f"{key}:state"
                    state_bytes = self.binary_redis.get(state_key)
                    
                    if state_bytes:
                        # Deserialize the state object
                        state = pickle.loads(state_bytes)
                        
                        # Load state into model
                        model.load_state(state)
                        logger.debug(
                            f"Loaded semantic cache match with similarity {best_similarity:.4f} "
                            f"for prompt hash {best_match[:8]}..."
                        )
                        return True
            
            # No suitable cache entry found
            return False
            
        except Exception as e:
            logger.error(f"Failed to load state from cache: {e}")
            return False
    
    def clear(self) -> None:
        """Clear all cache entries for this model"""
        try:
            # Get all keys with this prefix
            keys = self.redis.keys(f"{self.prefix}*") + self.redis.keys(f"{self.vector_prefix}*")
            
            if keys:
                # Delete all keys
                self.redis.delete(*keys)
                logger.info(f"Cleared {len(keys)} cache entries")
            
        except Exception as e:
            logger.error(f"Failed to clear cache: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        try:
            # Get all cache entry keys
            keys = self.redis.keys(f"{self.prefix}*")
            vector_keys = self.redis.keys(f"{self.vector_prefix}*")
            num_entries = len(keys) // 2  # Divide by 2 because each entry has data and state keys
            
            # Get Redis memory info
            memory_info = self.redis.info("memory")
            
            return {
                "entries": num_entries,
                "vectors": len(vector_keys),
                "used_memory_bytes": int(memory_info.get("used_memory", 0)),
                "used_memory_human": memory_info.get("used_memory_human", "unknown"),
                "maxmemory_bytes": int(memory_info.get("maxmemory", 0)),
                "maxmemory_human": memory_info.get("maxmemory_human", "unknown"),
                "maxmemory_policy": memory_info.get("maxmemory_policy", "unknown"),
            }
            
        except Exception as e:
            logger.error(f"Failed to get cache stats: {e}")
            return {
                "error": str(e),
                "entries": 0,
                "vectors": 0,
                "used_memory_bytes": 0,
                "used_memory_human": "error",
            }
    
    def __contains__(self, tokens: List[int]) -> bool:
        """
        Check if tokens are in cache (exact match only).
        
        Args:
            tokens: Tokenized prompt
            
        Returns:
            True if in cache, False otherwise
        """
        try:
            hash_str = self.compute_hash(tokens)
            key = self.get_key(hash_str)
            return self.redis.exists(key) > 0
        except Exception as e:
            logger.error(f"Error checking cache: {e}")
            return False
    
    def __getitem__(self, tokens: List[int]) -> Any:
        """
        Get state from cache by tokens.
        
        Args:
            tokens: Tokenized prompt
            
        Returns:
            Model state object
            
        Raises:
            KeyError: If tokens not in cache
        """
        hash_str = self.compute_hash(tokens)
        key = self.get_key(hash_str)
        
        if not self.redis.exists(key):
            raise KeyError(f"No cache entry for tokens")
        
        state_key = f"{key}:state"
        state_bytes = self.binary_redis.get(state_key)
        
        if not state_bytes:
            raise KeyError(f"Cache entry exists but state data is missing")
        
        # Deserialize the state object
        return pickle.loads(state_bytes)
    
    def __setitem__(self, tokens: List[int], state) -> None:
        """
        Set state in cache.
        
        Args:
            tokens: Tokenized prompt
            state: Model state object
        """
        try:
            hash_str = self.compute_hash(tokens)
            key = self.get_key(hash_str)
            
            # We need the prompt text for embedding, but we don't have it here
            # Just store with the hash data only, no embedding
            cache_data = {
                "hash": hash_str,
                "timestamp": time.time(),
            }
            
            # Serialize the state object
            state_bytes = pickle.dumps(state)
            
            # Store metadata in a hash
            pipe = self.redis.pipeline()
            pipe.hset(key, mapping=cache_data)
            
            # Store binary state data separately
            state_key = f"{key}:state"
            self.binary_redis.set(state_key, state_bytes)
            
            # Set expiration if TTL is configured
            if self.ttl > 0:
                pipe.expire(key, self.ttl)
                self.binary_redis.expire(state_key, self.ttl)
            
            # Execute pipeline
            pipe.execute()
            
        except Exception as e:
            logger.error(f"Failed to set cache item: {e}") 
