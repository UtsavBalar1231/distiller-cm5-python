"""
Processing layer for MCP client.
This file combines message, tool and prompt processors into a single module.
"""

import time
import json
import logging
from mcp import ClientSession
from datetime import datetime
from typing import List, Dict, Any, Union, Optional, Coroutine, Literal, Tuple

from pydantic import BaseModel, field_validator, model_validator

from distiller_cm5_python.utils.config import (
    MAX_MESSAGES_LENGTH,
    DEFAULT_SYSTEM_PROMPT,
    LOGGING_LEVEL,
)
from distiller_cm5_python.utils.distiller_exception import (
    UserVisibleError,
    LogOnlyError,
)

# Get logger instance for this module
logger = logging.getLogger(__name__)


def timestamp_to_time(timestamp):
    dt_object = datetime.fromtimestamp(timestamp)
    formatted_time = dt_object.strftime("%Y-%m-%d %H:%M:%S")
    return formatted_time


# --- Pydantic Models for Message Structure ---

DEFAULT_SYSTEM_MESSAGE = ''

ROLE = 'role'
CONTENT = 'content'
REASONING_CONTENT = 'reasoning_content'
NAME = 'name'

SYSTEM = 'system'
USER = 'user'
ASSISTANT = 'assistant'
FUNCTION = 'function' # Represents Tool Results internally
TOOL = 'tool' # API role for tool results

# Content Types (for potential future multi-modal use)
TEXT = 'text'
FILE = 'file'
IMAGE = 'image'
AUDIO = 'audio'
VIDEO = 'video'


class BaseModelCompatibleDict(BaseModel):
    """ Base model providing dictionary-like access and serialization control. """
    def __getitem__(self, item):
        try:
            return getattr(self, item)
        except AttributeError:
            raise KeyError(item)

    def __setitem__(self, key, value):
        setattr(self, key, value)

    def model_dump(self, **kwargs):
        if 'exclude_none' not in kwargs:
            kwargs['exclude_none'] = True
        # Rely on Pydantic's super().model_dump() to correctly serialize
        # nested models and basic types. Stringification of arguments for API
        # is handled in get_formatted_messages.
        dumped_data = super().model_dump(**kwargs)
        return dumped_data

    def model_dump_json(self, **kwargs):
        if 'exclude_none' not in kwargs:
            kwargs['exclude_none'] = True
        # Use the custom model_dump logic
        dict_repr = self.model_dump(**kwargs)
        return json.dumps(dict_repr, **kwargs)

    def get(self, key, default=None):
        return getattr(self, key, default)

    def __str__(self):
        # Exclude 'extra' from string representation for brevity if empty
        dump = self.model_dump()
        # Clean up extra field for printing
        if 'extra' in dump:
            if not dump['extra']:
                del dump['extra']
            # Avoid printing potentially large args in tool_calls within extra
            elif 'tool_calls' in dump['extra']:
                 dump['extra'] = {**dump['extra'], 'tool_calls': '[present]'}

        return f'{self.__class__.__name__}({dump})'

    def __repr__(self):
        return self.__str__()

    class Config:
        # Allow extra fields to be stored but not validated (useful for 'extra')
        extra = 'allow'


class FunctionCall(BaseModelCompatibleDict):
    """ Represents a function call requested by the assistant. """
    name: str
    # Arguments are often passed as a JSON string by LLMs, but can be dict internally
    arguments: Union[str, Dict[str, Any]]

    def __init__(self, name: str, arguments: Union[str, Dict[str, Any]]):
        super().__init__(name=name, arguments=arguments)


class ContentItem(BaseModelCompatibleDict):
    """ Represents a single piece of content within a message (e.g., text, image). """
    text: Optional[str] = None
    image: Optional[str] = None # Placeholder for image data/URI
    file: Optional[str] = None # Placeholder for file data/URI
    audio: Optional[Union[str, dict]] = None # Placeholder for audio data/URI
    video: Optional[Union[str, list]] = None # Placeholder for video data/URI

    def __init__(self,
                 text: Optional[str] = None,
                 image: Optional[str] = None,
                 file: Optional[str] = None,
                 audio: Optional[Union[str, dict]] = None,
                 video: Optional[Union[str, list]] = None):
        super().__init__(text=text, image=image, file=file, audio=audio, video=video)

    @model_validator(mode='after')
    def check_exclusivity(self):
        provided_fields = sum(1 for v in [self.text, self.image, self.file, self.audio, self.video] if v is not None)
        if provided_fields != 1:
            raise ValueError(f"Exactly one of '{TEXT}', '{IMAGE}', '{FILE}', '{AUDIO}', or '{VIDEO}' must be provided. Found {provided_fields}.")
        return self

    def get_type_and_value(self) -> Tuple[Literal['text', 'image', 'file', 'audio', 'video'], Any]:
        # Iterate through fields to find the non-None one
        for field_name in [TEXT, IMAGE, FILE, AUDIO, VIDEO]:
            value = getattr(self, field_name)
            if value is not None:
                return field_name, value
        raise ValueError("ContentItem is invalid; no value field is set.") # Should not happen if validator works

    @property
    def type(self) -> Literal['text', 'image', 'file', 'audio', 'video']:
        t, _ = self.get_type_and_value()
        return t

    @property
    def value(self) -> Any:
        _, v = self.get_type_and_value()
        return v

# Represents the structure expected by many LLM APIs for assistant tool calls
class ToolCall(BaseModelCompatibleDict):
    id: str
    type: Literal['function'] = 'function'
    function: FunctionCall

    def __init__(self, id: str, function: FunctionCall):
        super().__init__(id=id, function=function)


class Message(BaseModelCompatibleDict):
    """ Represents a single message in the conversation history using Pydantic. """
    role: str
    content: Union[str, List[ContentItem], None] # Allow None content, e.g., for assistant message with only tool calls
    # Optional fields based on role and context
    name: Optional[str] = None # Used for FUNCTION role (tool_call_id) or optionally USER role name
    # Tool calls specifically for assistant messages
    tool_calls: Optional[List[ToolCall]] = None
    # Tool call ID specifically for tool/function result messages
    tool_call_id: Optional[str] = None
    extra: Optional[Dict[str, Any]] = None # Flexible field for additional metadata

    def __init__(self,
                 role: str,
                 content: Union[str, List[ContentItem], None],
                 name: Optional[str] = None,
                 tool_calls: Optional[List[ToolCall]] = None,
                 tool_call_id: Optional[str] = None,
                 extra: Optional[Dict[str, Any]] = None,
                 **kwargs): # Allow capturing other potential fields into extra

        # Consolidate extra fields passed via kwargs
        if kwargs:
            if extra is None:
                extra = {}
            extra.update(kwargs)

        super().__init__(role=role,
                         content=content,
                         name=name,
                         tool_calls=tool_calls,
                         tool_call_id=tool_call_id,
                         extra=extra)

    @field_validator('role')
    def role_checker(cls, value: str) -> str:
        # Allow 'tool' initially for easier transition from old format, map later if needed
        allowed_roles = [USER, ASSISTANT, SYSTEM, FUNCTION, TOOL]
        if value not in allowed_roles:
            raise ValueError(f"Role '{value}' must be one of {', '.join(allowed_roles)}")
        # Map 'tool' role internally to 'function' for consistency
        if value == TOOL:
            return FUNCTION
        return value

    @field_validator('content')
    def content_validator(cls, value: Union[str, List[ContentItem], None]) -> Union[str, List[ContentItem], None]:
        if isinstance(value, list):
            if not value: # Empty list is allowed
                 return value
            if not all(isinstance(item, ContentItem) for item in value):
                raise ValueError("If content is a list, all items must be ContentItem objects.")
        # Allow string or None, but raise error for other types
        elif not isinstance(value, (str, type(None))):
             raise ValueError("Content must be a string, None, or a list of ContentItem objects.")
        return value

    @model_validator(mode='after')
    def check_role_specific_fields(self):
        # Tool calls should only exist for assistant messages
        if self.role != ASSISTANT and self.tool_calls is not None:
            raise ValueError(f"'tool_calls' field is only applicable for role '{ASSISTANT}'. Role is '{self.role}'.")

        # Tool call ID should only exist for function/tool messages
        if self.role != FUNCTION and self.tool_call_id is not None:
             raise ValueError(f"'tool_call_id' field is only applicable for role '{FUNCTION}' (or '{TOOL}'). Role is '{self.role}'.")

        # Function/tool messages must have a tool_call_id
        if self.role == FUNCTION and self.tool_call_id is None:
            raise ValueError(f"Messages with role '{FUNCTION}' (or '{TOOL}') must have a 'tool_call_id'.")

        # Assistant messages with tool_calls might have None or empty string content
        if self.role == ASSISTANT and self.tool_calls:
            if self.content is not None and not isinstance(self.content, (str, list)):
                 # This case should ideally be caught by content_validator, but double-checking
                 raise ValueError("Assistant message content must be string, list, or None.")
        elif self.content is None and self.role != ASSISTANT: # Only assistant can have None content (with tool calls)
             raise ValueError(f"Message content cannot be None for role '{self.role}'.")

        return self

# --- End Pydantic Models ---


class MessageProcessor:
    """Processor for managing conversation history and formatting messages for LLM providers."""

    def __init__(self):
        """Initialize the message processor"""
        super().__init__()  # No verbose parameter needed

        self.message_history: List[Message] = [] # Use Pydantic Message model
        self.session_start_time = timestamp_to_time(time.time()).replace(":", "-")

        # Check config for enabling debug message traffic dump
        self.save_debug_traffic = LOGGING_LEVEL == "DEBUG"
        self.debug_history_file = (
            f"debug_message_traffic_{self.session_start_time}.json"
        )

        logger.debug(
            f"MessageProcessor.__init__: Initialized with {'Enabled' if self.save_debug_traffic else 'Disabled'} mode"
        )

    def add_message(
        self,
        role: str,
        content: Union[str, List[ContentItem], None],
        metadata: Dict[str, Any] = None,
        tool_calls: Optional[List[Dict[str, Any]]] = None, # Raw tool calls for assistant
        tool_call_id: Optional[str] = None, # ID for tool results
    ) -> None:
        """Add a message to the conversation history using the Message model.

        Args:
            role: The role of the message sender (user, assistant, system, tool)
            content: The content of the message (string, list of ContentItems, or None)
            metadata: Optional metadata for the message (stored in 'extra')
            tool_calls: Raw tool call dictionaries (for assistant role only)
            tool_call_id: ID corresponding to the tool call (for tool role only)
        """

        # Control message length
        if len(self.message_history) >= MAX_MESSAGES_LENGTH:
            logger.warning(
                f"MessageProcessor.add_message: Message history is full. Removing oldest message."
            )
            # Traverse the list from the beginning, find the first message that is not of the 'system' role
            for i, msg in enumerate(self.message_history):
                if msg.role != SYSTEM:
                    # Delete the first non-system role message
                    del self.message_history[i]
                    break

        try:
            # Prepare Pydantic Message arguments
            message_args = {
                "role": role,
                "content": content,
                "tool_call_id": tool_call_id, # Will be validated based on role
                "extra": metadata or {},
            }

            # Process raw tool_calls for assistant messages into Pydantic ToolCall objects
            if role == ASSISTANT and tool_calls:
                parsed_tool_calls = []
                for tc in tool_calls:
                    try:
                        # Ensure arguments are parsed/handled correctly (assuming string or dict)
                        func_args = tc.get('function', {}).get('arguments')
                        if isinstance(func_args, str):
                            try:
                                # Attempt to parse if it looks like JSON, otherwise keep as string
                                if func_args.strip().startswith(("{", "[")):
                                    func_args = json.loads(func_args)
                            except json.JSONDecodeError:
                                pass # Keep as string if not valid JSON

                        pydantic_func = FunctionCall(
                            name=tc.get('function', {}).get('name', 'unknown_function'),
                            arguments=func_args
                        )
                        pydantic_tool_call = ToolCall(
                            id=tc.get('id', 'unknown_id'),
                            function=pydantic_func
                        )
                        parsed_tool_calls.append(pydantic_tool_call)
                    except Exception as e:
                        logger.error(f"Failed to parse tool call structure: {tc}. Error: {e}")
                        # Decide whether to skip this tool call or raise
                        continue
                message_args["tool_calls"] = parsed_tool_calls
                # Per API spec, content might be None/empty if tool_calls are present
                if not content and parsed_tool_calls:
                     message_args["content"] = None

            elif role == TOOL:
                # The 'tool' role from input is mapped to FUNCTION internally by Pydantic model
                 if not tool_call_id:
                     logger.warning("Attempted to add tool result message without tool_call_id. Discarding.")
                     return # Early exit, finally block will still execute

            # Attempt to create the Pydantic Message object
            try:
                message = Message(**message_args)
            except Exception as e: # Catches Pydantic validation errors for Message
                logger.error(f"Failed to create Pydantic Message object with args {message_args}: {e}")
                # Re-raise to indicate failure; finally block will execute before propagation.
                raise LogOnlyError(f"Internal error creating message object: {e}")

            # If message creation was successful, add to history
            self.message_history.append(message)

        finally:
            # This will be executed even if an early return happens (e.g., for TOOL role)
            # or if LogOnlyError is raised from the Message creation.
            if self.save_debug_traffic:
                self._save_debug_traffic()

    def set_system_message(self, content: str, metadata: Dict[str, Any] = None) -> None:
        """Set or replace the system message in the conversation history

        This method removes any existing system messages and adds a new one.

        Args:
            content: The content of the system message
            metadata: Optional metadata for the message
        """
        # Remove any existing system messages
        self.message_history = [
            msg for msg in self.message_history if msg.role != SYSTEM
        ]

        # Add the new system message
        metadata = metadata or {}
        if self.save_debug_traffic:
            metadata["message_type"] = "system_message_set"

        self.add_message(SYSTEM, content, metadata)

        logger.debug(
            f"MessageProcessor.set_system_message: System message set: {str(content)[:50]}..."
        )

    def add_tool_call(self, tool_call: Dict[str, Any]) -> None:
        """Add a tool call message to the assistant's last turn. Returns None."""
        try:
            tool_name = (
                tool_call.get("function", {}).get("name", "")
                if "function" in tool_call
                else tool_call.get("name", "")
            )
            tool_args = (
                tool_call.get("function", {}).get("arguments", {})
                if "function" in tool_call
                else tool_call.get("arguments", {})
            )
            tool_call_id = tool_call.get("id", tool_name)

            if not self.message_history or self.message_history[-1].role != ASSISTANT:
                logger.error("add_tool_call: Cannot add tool call. Last message is not from the assistant.")
                # Raise to indicate failure; finally block will execute before propagation.
                raise LogOnlyError("Attempted to add tool call when last message was not from assistant.")

            assistant_message = self.message_history[-1]

            # Prepare the single tool call in the Pydantic structure
            try:
                if isinstance(tool_args, str):
                    try:
                        # Attempt to parse if it looks like JSON, otherwise keep as string
                        if tool_args.strip().startswith(("{", "[")):
                            tool_args = json.loads(tool_args)
                    except json.JSONDecodeError:
                        pass # Keep as string if not valid JSON
                pydantic_func = FunctionCall(name=tool_name, arguments=tool_args)
                pydantic_tool_call = ToolCall(id=tool_call_id, function=pydantic_func)
            except Exception as e: # Catches Pydantic validation errors for FunctionCall/ToolCall
                logger.error(f"Failed to create Pydantic ToolCall structure: {e}")
                # Re-raise to indicate failure; finally block will execute before propagation.
                raise LogOnlyError(f"Internal error creating ToolCall object: {e}")

            if assistant_message.tool_calls is None:
                assistant_message.tool_calls = []

            assistant_message.tool_calls.append(pydantic_tool_call)

            if not assistant_message.content:
                 assistant_message.content = None
            # Implicit return None if successful
        finally:
            if self.save_debug_traffic:
                self._save_debug_traffic()

    def add_tool_result(self, tool_call: Dict[str, Any], result: str) -> None:
        """Add a tool result message to the history. Returns None."""

        tool_name = (
            tool_call.get("function", {}).get("name", "")
            if "function" in tool_call
            else tool_call.get("name", "")
        )
        tool_args = (
            tool_call.get("function", {}).get("arguments", {})
            if "function" in tool_call
            else tool_call.get("arguments", {})
        )
        tool_call_id = tool_call.get("id", tool_name)

        metadata = {
            "tool_name": tool_name,
            "tool_call_id": tool_call_id,
            "tool_args": tool_args,
            "is_tool_result": True,
            "message_type": "tool_result",
        }

        self.add_message(FUNCTION, result, metadata)

        # If debug traffic dump is enabled, track detailed message traffic
        if self.save_debug_traffic:
            self._save_debug_traffic()

        # Truncate result for logging
        log_result_snippet = str(result)[:100] + (
            "..." if len(str(result)) > 100 else ""
        )
        logger.debug(
            f"MessageProcessor.add_tool_result: Added result for '{tool_name}'. Content: '{log_result_snippet}'"
        )

        # Return None as the event is not used by the caller
        return None

    def get_formatted_messages(
        self,
    ) -> List[Dict[str, Union[str, List[Dict[str, Any]]]]]:
        """Get the message history formatted for LLM API

        Returns:
            List of formatted messages
        """

        formatted_messages = []

        for message in self.message_history:
            role = message.role
            content = message.content
            tool_calls = message.tool_calls if message.tool_calls else []
            metadata = message.extra
            
            # Handle tool results with special formatting
            if role == TOOL and metadata.get("is_tool_result", False):
                tool_name = metadata.get("tool_name", "unknown_tool")
                tool_call_id = metadata.get("tool_call_id", tool_name)
                formatted_messages.append(
                    {
                        "role": TOOL,
                        "type": "function_call_output",
                        "tool_call_id": tool_call_id,
                        "content": content,
                    }
                )

            elif role in [USER, SYSTEM] or (
                role == ASSISTANT and len(tool_calls) == 0
            ):
                formatted_messages.append({"role": role, "content": content})
            elif role == ASSISTANT:
                # Standard message formatting
                formatted_messages.append(
                    {"role": role, "content": content, "tool_calls": tool_calls}
                )

        logger.debug(
            f"MessageProcessor.get_formatted_messages: {len(formatted_messages)} messages returned"
        )
        return formatted_messages

    def _save_debug_traffic(self) -> None:
        """Save the debug message traffic to a JSON file"""
        # This method is now only called if self.save_debug_traffic is True
        if not self.save_debug_traffic:
            # This warning should not occur due to the finally blocks
            logger.warning("_save_debug_traffic called unexpectedly when disabled.")
            return

        try:
            # Get the messages formatted exactly as they would be sent to the API
            api_formatted_messages = self.get_formatted_messages()

            debug_data = {
                "session_start": self.session_start_time,
                "debug_traffic_api_format": api_formatted_messages, # Just save the API format
            }

            with open(self.debug_history_file, "w", encoding="utf-8") as f:
                # Dump the API-formatted data
                json.dump(debug_data, f, indent=2, default=str) # default=str for safety

            logger.debug(
                f"MessageProcessor._save_debug_traffic: Debug traffic saved to {self.debug_history_file}"
            )
        except Exception as e:
            # Catch any general exception during formatting or dumping
            logger.error(f"MessageProcessor._save_debug_traffic: Failed to save debug traffic: {e}")
            # Log the data we attempted to dump if possible
            if 'api_formatted_messages' in locals():
                try:
                    import pprint
                    logger.error(f"Data attempted to dump:\n{pprint.pformat(api_formatted_messages)}")
                except Exception as log_e:
                     logger.error(f"Could not log the data structure due to: {log_e}")

    def process(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Process messages and return formatted messages for LLM API
        Args:
            messages: A list of messages to process
        Returns:
            Formatted messages for LLM API
        """
        # This method takes raw input messages (dicts), processes them into internal
        # Pydantic Message objects, and returns the API-ready formatted list.
        for message_dict in messages:
            # Extract data from the input dictionary
            role = message_dict.get("role", USER) # Default to USER if role missing
            content = message_dict.get("content", None) # Allow None content for input
            metadata = message_dict.get("metadata", {})
            # Potentially extract tool_calls or tool_call_id if they can appear in raw input
            raw_tool_calls = message_dict.get("tool_calls")
            tool_call_id = message_dict.get("tool_call_id") # Relevant if input can contain tool results

            # Use add_message to convert dict to internal Pydantic Message object and add to history
            try:
                self.add_message(
                    role=role,\
                    content=content,\
                    metadata=metadata,\
                    tool_calls=raw_tool_calls, # Pass raw tool calls if present
                    tool_call_id=tool_call_id  # Pass tool_call_id if present
                )
            except Exception as e:
                 logger.error(f"Error processing input message dict: {message_dict}. Error: {e}")
                 # Decide whether to skip this message or raise error
                 continue

        # Return the messages formatted for the API call
        return self.get_formatted_messages()


class ToolProcessor:
    """Manages tool execution and formatting for LLM providers"""

    def __init__(self, session: Optional[ClientSession] = None):
        """Initialize the tool processor

        Args:
            session: MCP client session
        """
        self.session = session
        self.tools = []

        logger.debug(
            f"ToolProcessor.__init__: Initialized with session: {session is not None}"
        )

    async def refresh_capabilities(self) -> None:
        """Refresh available tools from the MCP server"""
        if not self.session:
            logger.error("ToolProcessor.refresh_capabilities: No session available")
            raise UserVisibleError(
                "No connected Mcp server session available, please check Mcp server connection"
            )

        logger.debug(f"ToolProcessor.refresh_capabilities: Refreshing tools")

        # Get available tools
        try:
            tools_response = await self.session.list_tools()
            self.tools = tools_response.tools

            logger.debug(
                f"ToolProcessor.refresh_capabilities: Got {len(self.tools)} tools"
            )
            for tool in self.tools:
                logger.debug(f"ToolProcessor.refresh_capabilities: Tool: {tool.name}")

        except Exception as e:
            logger.error(
                f"ToolProcessor.refresh_capabilities: Failed to refresh tools: {e}"
            )
            self.tools = []

    def format_tools(self) -> List[Dict[str, Any]]:
        """Format all available tools from server into the shape expected by LLM models"""
        if not self.session or not hasattr(self, "tools") or not self.tools:
            return []

        formatted_tools = []

        for tool in self.tools:
            # Check if tool is a dictionary or an object with attributes
            if isinstance(tool, dict):
                name = tool.get("name", "")
                description = tool.get("description", "")
                parameters = tool.get("inputSchema", {})
            else:
                # Assume it's an object with attributes
                name = getattr(tool, "name", "")
                description = getattr(tool, "description", "")
                parameters = getattr(tool, "inputSchema", {})

            formatted_tool = {
                "type": "function",
                "function": {
                    "name": name,
                    "description": description,
                    "parameters": parameters,
                },
            }
            formatted_tools.append(formatted_tool)

        logger.debug(
            f"ToolProcessor.format_tools: Return {len(formatted_tools)} formatted tools"
        )
        return formatted_tools

    async def execute_tool_call_async(self, tool_call: Dict[str, Any]) -> str:
        """Execute the tool call asynchronously

        Args:
            tool_call: The tool call to execute

        Returns:
            The result of the tool execution
        """
        if not self.session:
            logger.error(
                "ToolProcessor.execute_tool_call_async: No MCP session available"
            )
            return json.dumps({"error": "MCP session not available"})

        tool_name = tool_call.get("function", {}).get("name", "")
        args = tool_call.get("function", {}).get("arguments", {})

        # Extract arguments; could be string (JSON) or dict
        if isinstance(args, str):
            try:
                # Try to parse as JSON if it's a string
                import json

                args = json.loads(args.strip())
            except:
                # If fails, use as is - will likely cause issues,
                # but better than failing entirely
                pass

        logger.debug(
            f"ToolProcessor.execute_tool_call_async: Executing {tool_name} with args: {args}"
        )

        try:
            # Call the tool via the MCP session
            result = (
                await self.session.call_tool(tool_name, args)
                if args
                else await self.session.call_tool(tool_name, None)
            )

            # Format the result for display
            formatted_result = self._format_tool_result(result)
            return formatted_result

        except Exception as e:
            error_msg = f"Error executing tool {tool_name}: {e}"
            logger.error(error_msg)
            return json.dumps({"error": f"Tool execution failed: {str(e)}"})

    def _format_tool_result(self, result) -> str:
        """Format a tool result into a string representation.

        Args:
            result: The result object returned from the MCP server

        Returns:
            A string representation of the result
        """
        # Check if result has a 'content' attribute that is a list (common MCP format)
        if hasattr(result, "content") and isinstance(result.content, list):
            result_text = []
            for item in result.content:
                # Handle TextContent objects or dicts with 'text' key
                if hasattr(item, "text"):
                    result_text.append(item.text)
                elif isinstance(item, dict) and "text" in item:
                    result_text.append(item["text"])
                else:
                    # Fallback for other item types within the content list
                    result_text.append(str(item))
            return "\n".join(result_text)

        # Handle if the result itself is a list
        elif isinstance(result, list):
            result_text = []
            for item in result:
                if isinstance(item, dict):
                    # Handle content object
                    content_type = item.get("type", "unknown")

                    if content_type == "text":
                        # Extract text content
                        text = item.get("text", "")
                        result_text.append(text)
                    elif "content" in item:
                        # Extract generic content
                        result_text.append(str(item["content"]))
                    else:
                        # Just extract whole item as string
                        result_text.append(str(item))
                else:
                    # Just add as string
                    result_text.append(str(item))

            # Join text items with newlines
            return "\n".join(result_text)
        elif isinstance(result, dict):
            # Try to extract content from dict
            if "text" in result:
                return result["text"]
            elif "content" in result:
                return str(result["content"])
            else:
                # Format as JSON
                import json

                try:
                    return json.dumps(result, indent=2)
                except:
                    return str(result)
        else:
            # Return as string
            return str(result)


class PromptProcessor:
    """Processor for generating system prompts with optional tool information."""

    def __init__(self):
        """Initialize the prompt processor"""

        # Default user system prompt
        self.default_user_prompt = DEFAULT_SYSTEM_PROMPT

        logger.debug(
            f"PromptProcessor.__init__: Initialized with default system prompt, {self.default_user_prompt}"
        )

    async def format_prompts(self, session):
        """Format the prompts for the MCP server"""
        prompts = []
        available_prompts = await session.list_prompts()
        logger.debug(
            f"PromptProcessor.format_prompts: Available prompts: {available_prompts}"
        )
        for prompt in available_prompts.prompts:
            prompt_result = await session.get_prompt(
                prompt.name, prompt.arguments if prompt.arguments else {}
            )
            formatted_prompt = {
                "name": prompt.name,
                "description": prompt_result.description,
                "messages": [
                    {"role": msg.role, "content": msg.content.text}
                    for msg in prompt_result.messages
                ],
            }
            prompts.append(formatted_prompt)
        return prompts

    def generate_system_prompt(self, additional_prompt: Optional[str] = None) -> str:
        """Generate a complete system prompt

        Args:
            additional_prompt: Optional custom user prompt to add to the default system prompt

        Returns:
            A formatted system prompt
        """
        prompt = self.default_user_prompt
        if additional_prompt:
            prompt += "\n\n" + additional_prompt

        logger.debug(
            f"PromptProcessor.generate_system_prompt: Generated system prompt: {prompt}"
        )
        return prompt


# Export the classes for easy importing
__all__ = ["MessageProcessor", "ToolProcessor", "PromptProcessor"]
