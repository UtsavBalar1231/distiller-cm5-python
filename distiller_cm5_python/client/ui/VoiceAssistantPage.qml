import "Components"
import QtQuick 2.15
import QtQuick.Controls 2.15
import QtQuick.Layouts 1.15

PageBase {
    id: voiceAssistantPage
    
    pageName: "Voice Assistant"

    // Use property setter pattern to ensure component updates
    property string _serverName: "MCP Server"
    property string serverName: _serverName
    property bool isListening: false
    property bool isProcessing: false
    property string statusText: "Ready"
    property string inputBuffer: ""
    property var focusableItems: []
    property var previousFocusedItem: null

    signal selectNewServer()

    onServerNameChanged: {
        _serverName = serverName;
    }
    
    // Collect all focusable items on this page
    function collectFocusItems() {
        console.log("VoiceAssistantPage: Collecting focusable items");
        focusableItems = []
        
        // Add buttons from InputArea
        if (inputArea) {
            console.log("InputArea found, adding buttons");
            
            // Access buttons through the exposed properties
            if (inputArea.settingsButton && inputArea.settingsButton.navigable) {
                console.log("Adding settings button to focusable items");
                focusableItems.push(inputArea.settingsButton)
            }
            
            if (inputArea.voiceButton && inputArea.voiceButton.navigable) {
                console.log("Adding voice button to focusable items");
                focusableItems.push(inputArea.voiceButton)
            }
            
            if (inputArea.sendButton && inputArea.sendButton.navigable) {
                console.log("Adding send button to focusable items");
                focusableItems.push(inputArea.sendButton)
            }
        } else {
            console.log("InputArea not found or not fully initialized yet");
        }
        
        // Add server select button in header if present
        if (header && header.serverSelectButton && header.serverSelectButton.navigable) {
            console.log("Adding server select button to focusable items");
            focusableItems.push(header.serverSelectButton)
        }
        
        console.log("Total focusable items: " + focusableItems.length);
        
        // Initialize focus manager with conversation view for scrolling
        FocusManager.initializeFocusItems(focusableItems, conversationView)
    }
    
    // Function to ensure input area buttons are focused if nothing else is
    function ensureFocusableItemsHaveFocus() {
        console.log("Ensuring something has focus");
        // If no focus or focus index is -1, reset focus to input area
        if (FocusManager.currentFocusIndex < 0 || FocusManager.currentFocusItems.length === 0) {
            console.log("Focus needs to be reset");
            // Re-collect focus items to ensure they're registered
            collectFocusItems();
            
            // Set focus to one of the input buttons
            if (inputArea && inputArea.voiceButton && inputArea.voiceButton.navigable) {
                console.log("Setting focus to voice button");
                FocusManager.setFocusToItem(inputArea.voiceButton);
            } else if (inputArea && inputArea.settingsButton && inputArea.settingsButton.navigable) {
                console.log("Setting focus to settings button");
                FocusManager.setFocusToItem(inputArea.settingsButton);
            } else if (focusableItems.length > 0) {
                console.log("Setting focus to first item in list");
                FocusManager.setFocusToItem(focusableItems[0]);
            }
        }
    }

    Component.onCompleted: {
        // Collect focusable items after component is fully loaded
        console.log("VoiceAssistantPage completed, scheduling focus collection");
        Qt.callLater(collectFocusItems);
    }

    // Add a timer to ensure focus items are collected after everything is fully loaded
    Timer {
        id: focusInitTimer
        interval: 500
        running: true
        repeat: false
        onTriggered: {
            console.log("Focus init timer triggered");
            collectFocusItems();
            
            // Set initial focus to voice button
            if (inputArea && inputArea.voiceButton && inputArea.voiceButton.navigable) {
                FocusManager.setFocusToItem(inputArea.voiceButton);
            }
        }
    }

    // Connect to bridge ready signal
    Connections {
        target: bridge
        
        function onBridgeReady() {
            // Initialize conversation when bridge is ready
            if (conversationView) {
                conversationView.updateModel(bridge.get_conversation());
            }
        }
    }

    // Header area with server name and status
    VoiceAssistantPageHeader {
        id: header

        anchors.top: parent.top
        anchors.left: parent.left
        anchors.right: parent.right
        height: 70 // Increased height to accommodate wrapping status text
        serverName: _serverName
        statusText: voiceAssistantPage.statusText
        isConnected: bridge && bridge.ready ? bridge.isConnected : false
        
        onServerSelectClicked: {
            // Store the currently focused item before showing dialog
            previousFocusedItem = FocusManager.currentFocusItems[FocusManager.currentFocusIndex];
            confirmServerChangeDialog.open();
        }
    }

    // Confirmation dialog for server change 
    AppDialog {
        id: confirmServerChangeDialog

        dialogTitle: "Change Server"
        message: "Are you sure you want to change servers? Current conversation will be lost."
        
        // Configure the standard buttons 
        standardButtonTypes: DialogButtonBox.Yes | DialogButtonBox.No
        
        // Button text customization
        yesButtonText: "Proceed"
        noButtonText: "Cancel"
        
        // Use accent color for the positive button
        acceptButtonColor: ThemeManager.accentColor
        
        onAccepted: {
            // Disconnect from current server
            if (bridge && bridge.ready) {
                bridge.disconnectFromServer();
            }
            // Go back to server selection
            voiceAssistantPage.selectNewServer();
        }
        
        onRejected: {
            // Restore focus to the previously focused item when canceling
            restoreFocusTimer.start();
        }
        
        // Handle dialog closure
        onClosed: {
            // If dialog is rejected (Cancel pressed), restore focus
            if (!visible && !accepted) {
                restoreFocusTimer.start();
            }
        }
    }

    // Conversation display area
    ConversationView {
        id: conversationView

        anchors.top: header.bottom
        anchors.left: parent.left
        anchors.right: parent.right
        anchors.bottom: inputArea.top
        anchors.bottomMargin: 4 // Add a gap between conversation and input area
        anchors.margins: ThemeManager.spacingNormal

        // Force model refresh when conversation changes
        Component.onCompleted: {
            if (bridge && bridge.ready) {
                updateModel(bridge.get_conversation());
            } else {
                updateModel([]);
            }
        }

        Connections {
            target: bridge && bridge.ready ? bridge : null
            
            function onConversationChanged() {
                conversationView.updateModel(bridge.get_conversation());
            }
            
            function onMessageReceived(message, timestamp) {
                isProcessing = false;
                isListening = false;
                statusText = "Ready";
                
                // Explicitly clear and reset the input area
                inputBuffer = "";
                
                // Delay turning off response mode slightly to ensure the final message is rendered
                responseEndTimer.start();
            }
            
            function onListeningStarted() {
                isListening = true;
                statusText = "Listening...";
                messageToast.showMessage("Listening...", 1500);
            }
            
            function onListeningStopped() {
                isListening = false;
                statusText = "Processing...";
                isProcessing = true;
                // Set response in progress to lock scrolling
                conversationView.setResponseInProgress(true);
            }
            
            function onErrorOccurred(errorMessage) {
                // Log the error to console for developer debugging
                console.error("Error occurred in bridge: " + errorMessage);
                
                // Show error toast with appropriate duration based on message length
                var displayDuration = Math.max(3000, Math.min(errorMessage.length * 75, 8000));
                messageToast.showMessage("Error: " + errorMessage, displayDuration);
                
                // Update UI state
                isProcessing = false;
                isListening = false;
                statusText = "Ready";
                
                // Enable scrolling on error
                conversationView.setResponseInProgress(false);
                
                // If error is related to connection, suggest reconnecting
                if (errorMessage.toLowerCase().includes("connect") || 
                    errorMessage.toLowerCase().includes("server") ||
                    errorMessage.toLowerCase().includes("timeout")) {
                    // Show reconnection dialog after a brief delay
                    reconnectionTimer.start();
                }
            }
            
            function onStatusChanged(newStatus) {
                statusText = newStatus;
            }
        }
    }

    // Message toast
    MessageToast {
        id: messageToast

        anchors.horizontalCenter: parent.horizontalCenter
        anchors.bottom: inputArea.top
        anchors.bottomMargin: ThemeManager.spacingNormal
    }

    // Full input area with buttons row
    InputArea {
        id: inputArea
        
        anchors.left: parent.left
        anchors.right: parent.right
        anchors.bottom: parent.bottom
        anchors.margins: 8
        
        isListening: voiceAssistantPage.isListening
        isProcessing: voiceAssistantPage.isProcessing
        
        onTextSubmitted: function(text) {
            if (bridge && bridge.ready && bridge.isConnected && !isProcessing) {
                inputBuffer = text;
                if (bridge.sendTextMessage(text)) {
                    isProcessing = true;
                    statusText = "Processing...";
                    // Set response in progress to lock scrolling
                    conversationView.setResponseInProgress(true);
                }
            }
        }
        
        onVoiceToggled: function(listening) {
            if (bridge && bridge.ready && bridge.isConnected && !isProcessing) {
                if (listening) {
                    bridge.startListening();
                } else {
                    bridge.stopListening();
                }
            }
        }
        
        onSettingsClicked: {
            // Navigate to the settings page using the application-defined function
            if (mainWindow && typeof mainWindow.pushSettingsPage === "function") {
                mainWindow.pushSettingsPage();
            }
        }
    }
    
    // Reconnection suggestion timer
    Timer {
        id: reconnectionTimer
        
        interval: 500
        repeat: false
        running: false
        
        onTriggered: {
            // Save the current focus before showing dialog
            previousFocusedItem = FocusManager.currentFocusItems[FocusManager.currentFocusIndex];
            // Show reconnection dialog
            confirmServerChangeDialog.open();
        }
    }
    
    // Response end timer to delay turning off response in progress mode
    Timer {
        id: responseEndTimer
        
        interval: 300
        repeat: false
        running: false
        
        onTriggered: {
            // Turn off response in progress mode
            conversationView.setResponseInProgress(false);
        }
    }

    // Timer to restore focus after dialog is closed
    Timer {
        id: restoreFocusTimer
        interval: 100
        repeat: false
        running: false
        
        onTriggered: {
            console.log("Restore focus timer triggered");
            // First make sure focus items are properly initialized
            collectFocusItems();
            
            // Log the state for debugging
            console.log("Current focus index: " + FocusManager.currentFocusIndex);
            console.log("Focusable items count: " + focusableItems.length);
            console.log("Has previous focus item: " + (previousFocusedItem !== null));
            
            // Then try to restore focus to the previous item or a default item
            if (previousFocusedItem && previousFocusedItem.navigable) {
                console.log("Restoring focus to previous item");
                FocusManager.setFocusToItem(previousFocusedItem);
            } else if (inputArea && inputArea.voiceButton && inputArea.voiceButton.navigable) {
                console.log("Setting focus to voice button");
                FocusManager.setFocusToItem(inputArea.voiceButton);
            } else if (focusableItems.length > 0) {
                console.log("Setting focus to first non-header button");
                // Default to the first button that's not the header button
                for (var i = 0; i < focusableItems.length; i++) {
                    if (focusableItems[i] !== header.serverSelectButton) {
                        FocusManager.setFocusToItem(focusableItems[i]);
                        break;
                    }
                }
            } else {
                console.error("No focusable items available to restore focus");
            }
            
            // Double-check that something has focus
            focusCheckTimer.start();
        }
    }
    
    // Timer to verify focus state after initial restoration
    Timer {
        id: focusCheckTimer
        interval: 200
        repeat: false
        running: false
        onTriggered: {
            ensureFocusableItemsHaveFocus();
        }
    }
}

