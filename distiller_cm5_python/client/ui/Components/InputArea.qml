// import QtGraphicalEffects 1.15  // Commented out in case it's not available

import QtQuick 2.15
import QtQuick.Controls 2.15
import QtQuick.Layouts 1.15
import Components 1.0

Rectangle {
    // No borders on the main rectangle

    id: inputArea

    // Dynamic height based on text content with min/max constraints
    property int minHeight: 80
    property int maxHeight: 160
    property bool isListening: false
    property bool isProcessing: false
    property bool compact: true

    // Expose buttons as properties
    property alias settingsButton: settingsButton
    property alias voiceButton: voiceButton
    property alias sendButton: sendButton

    signal textSubmitted(string text)
    signal voiceToggled(bool listening)
    signal settingsClicked()

    function clearInput() {
        textInput.clear();
    }

    function getText() {
        return textInput.text.trim();
    }

    // Add a new function to reset all input state
    function resetState() {
        isProcessing = false;
        textInput.readOnly = false;
        textInput.focus = true;
    }

    color: ThemeManager.backgroundColor
    height: Math.min(maxHeight, Math.max(minHeight, inputLayout.implicitHeight + 20))
    z: 10 // Ensure this is always on top

    // Hint text that appears above the input field
    Text {
        id: hintText

        visible: !isListening && (textInput.text.length > 30 || textInput.text.indexOf("\n") >= 0)
        anchors.right: parent.right
        anchors.bottom: inputLayout.top
        anchors.rightMargin: 12
        anchors.bottomMargin: 2
        text: "Shift+Enter for new line"
        font.pixelSize: 10
        color: ThemeManager.textColor
        opacity: 0.6

        // Add a subtle background to improve visibility
        Rectangle {
            z: -1
            anchors.fill: parent
            anchors.margins: -3
            color: ThemeManager.backgroundColor
            opacity: 0.8
            radius: 3
        }

    }

    // Listening hint that appears during microphone input
    Text {
        id: listeningHint

        visible: isListening
        anchors.horizontalCenter: parent.horizontalCenter
        anchors.bottom: inputLayout.top
        anchors.bottomMargin: 2
        text: "Listening..."
        font.pixelSize: FontManager.fontSizeNormal
        font.family: FontManager.primaryFontFamily
        color: ThemeManager.textColor
        opacity: 0.9

        // Add a subtle background to improve visibility
        Rectangle {
            z: -1
            anchors.fill: parent
            anchors.margins: -6
            color: ThemeManager.backgroundColor
            opacity: 0.8
            radius: 3
        }

    }

    // Simple grid layout with fixed proportions
    GridLayout {
        id: inputLayout

        anchors.fill: parent
        anchors.margins: 8
        rowSpacing: 4
        columnSpacing: 4
        columns: 1
        rows: 2

        // Input field - simple rectangle with text area
        Rectangle {
            id: inputField

            Layout.fillWidth: true
            Layout.fillHeight: true
            Layout.minimumHeight: 36
            color: ThemeManager.backgroundColor
            border.color: isListening ? "transparent" : ThemeManager.borderColor
            border.width: isListening ? 0 : ThemeManager.borderWidth
            radius: ThemeManager.borderRadius
            // Visual feedback when disabled
            opacity: inputArea.isProcessing ? 0.7 : 1.0

            // Stack content that switches between text area and visualizer
            Item {
                anchors.fill: parent

                // Text input scroll view
                AppScrollView {
                    id: scrollView

                    anchors.fill: parent
                    anchors.margins: 0
                    visible: !isListening && !isProcessing
                    showEdgeEffects: false
                    contentHeight: textInput.implicitHeight

                    TextArea {
                        id: textInput

                        anchors.fill: parent
                        anchors.margins: 4
                        wrapMode: TextArea.Wrap
                        font: FontManager.normal
                        color: ThemeManager.textColor
                        placeholderText: "Type your message here..."
                        placeholderTextColor: ThemeManager.secondaryTextColor
                        background: null
                        readOnly: inputArea.isProcessing || inputArea.isListening
                        // Allow vertical growth but limit it
                        onTextChanged: {
                            // Update implicitHeight when text changes
                            inputArea.implicitHeight = Math.min(inputArea.maxHeight, Math.max(inputArea.minHeight, inputLayout.implicitHeight + 20));
                        }
                        Keys.onReturnPressed: function(event) {
                            if (event.modifiers & Qt.ShiftModifier) {
                                // Allow shift+return for line breaks
                                event.accepted = false;
                            } else if (text.trim() !== "") {
                                sendButton.clicked();
                                event.accepted = true;
                            }
                        }
                    }

                }

                // Audio visualizer component
                AudioVisualizer {
                    id: audioVisualizer

                    anchors.fill: parent
                    anchors.margins: 4
                    visible: isListening
                    isActive: isListening
                }

            }

        }

        // Minimalist button row
        Item {
            id: buttonRow

            Layout.fillWidth: true
            Layout.preferredHeight: 40

            // Consistent button size
            property int buttonSize: 40
            property int borderWidth: 2

            Row {
                id: leftButtonsRow

                anchors.left: parent.left
                anchors.verticalCenter: parent.verticalCenter
                spacing: 12

                // Settings button
                RoundButton {
                    id: settingsButton

                    width: buttonRow.buttonSize
                    height: buttonRow.buttonSize
                    flat: true
                    property bool navigable: true
                    property bool isActiveItem: false
                    onClicked: inputArea.settingsClicked()

                    background: Rectangle {
                        color: "transparent"
                        antialiasing: true
                        
                        // Crisp focus border
                        Rectangle {
                            anchors.fill: parent
                            anchors.margins: settingsButton.isActiveItem ? 0 : -buttonRow.borderWidth
                            color: "transparent"
                            border {
                                width: settingsButton.isActiveItem ? buttonRow.borderWidth : 0
                                color: ThemeManager.accentColor
                            }
                            radius: width / 2
                            antialiasing: true
                            opacity: settingsButton.isActiveItem ? 1.0 : 0
                        }
                    }

                    contentItem: Item {
                        anchors.fill: parent
                        
                        // Simple hover highlight
                        Rectangle {
                            visible: settingsButton.hovered || settingsButton.pressed
                            anchors.fill: parent
                            radius: width / 2
                            color: ThemeManager.buttonColor
                            opacity: 0.15
                            antialiasing: true
                        }

                        Text {
                            text: "⚙"  // Gear icon as text
                            font.pixelSize: parent.width / 2
                            font.family: FontManager.primaryFontFamily
                            color: ThemeManager.textColor
                            horizontalAlignment: Text.AlignHCenter
                            verticalAlignment: Text.AlignVCenter
                            anchors.centerIn: parent
                            opacity: settingsButton.isActiveItem ? 1.0 : (settingsButton.hovered ? 1 : 0.7)
                        }
                    }
                }

                // Voice button
                RoundButton {
                    id: voiceButton

                    width: buttonRow.buttonSize
                    height: buttonRow.buttonSize
                    flat: true
                    checkable: true
                    property bool navigable: true
                    property bool isActiveItem: false
                    checked: inputArea.isListening
                    onClicked: {
                        inputArea.voiceToggled(checked);
                    }

                    background: Rectangle {
                        color: voiceButton.checked ? ThemeManager.subtleColor : "transparent"
                        antialiasing: true
                        
                        // Crisp focus border
                        Rectangle {
                            anchors.fill: parent
                            anchors.margins: voiceButton.isActiveItem ? 0 : -buttonRow.borderWidth
                            color: "transparent"
                            border {
                                width: voiceButton.isActiveItem ? buttonRow.borderWidth : 0
                                color: ThemeManager.accentColor
                            }
                            radius: width / 2
                            antialiasing: true
                            opacity: voiceButton.isActiveItem ? 1.0 : 0
                        }
                    }

                    contentItem: Item {
                        anchors.fill: parent
                        
                        // Simple hover highlight
                        Rectangle {
                            visible: voiceButton.hovered || voiceButton.pressed
                            anchors.fill: parent
                            radius: width / 2
                            color: ThemeManager.buttonColor
                            opacity: 0.15
                            antialiasing: true
                        }

                        OptimizedImage {
                            id: micIcon
                            source: {
                                if (!voiceButton.enabled)
                                    return ThemeManager.darkMode ? "../images/icons/dark/microphone-empty.svg" : "../images/icons/microphone-empty.svg";

                                if (voiceButton.checked) {
                                    if (inputArea.isProcessing)
                                        return ThemeManager.darkMode ? "../images/icons/dark/microphone-filled.svg" : "../images/icons/microphone-filled.svg";

                                    return ThemeManager.darkMode ? "../images/icons/dark/microphone-half.svg" : "../images/icons/microphone-half.svg";
                                }
                                return ThemeManager.darkMode ? "../images/icons/dark/microphone-empty.svg" : "../images/icons/microphone-empty.svg";
                            }
                            sourceSize.width: 22
                            sourceSize.height: 22
                            width: 22
                            height: 22
                            anchors.centerIn: parent
                            fillMode: Image.PreserveAspectFit
                            opacity: voiceButton.checked ? 1 : (voiceButton.hovered ? 0.9 : 0.7)
                            fadeInDuration: 0
                            showPlaceholder: false
                        }

                        // Simple indicator for listening state
                        Rectangle {
                            visible: voiceButton.checked && !inputArea.isProcessing
                            anchors.centerIn: parent
                            width: parent.width - 4
                            height: parent.height - 4
                            radius: width / 2
                            color: "transparent"
                            border.width: 1
                            border.color: ThemeManager.accentColor
                            opacity: 0.5
                            antialiasing: true
                        }
                    }
                }
            }

            // Send button
            RoundButton {
                id: sendButton

                anchors.right: parent.right
                anchors.verticalCenter: parent.verticalCenter
                width: buttonRow.buttonSize
                height: buttonRow.buttonSize
                flat: true
                property bool navigable: true
                property bool isActiveItem: false
                enabled: textInput.text.trim() !== ""
                onClicked: {
                    if (textInput.text.trim() !== "") {
                        inputArea.textSubmitted(textInput.text.trim());
                        textInput.clear();
                    }
                }

                background: Rectangle {
                    color: "transparent"
                    antialiasing: true
                    
                    // Crisp focus border
                    Rectangle {
                        anchors.fill: parent
                        anchors.margins: sendButton.isActiveItem ? 0 : -buttonRow.borderWidth
                        color: "transparent"
                        border {
                            width: sendButton.isActiveItem ? buttonRow.borderWidth : 0
                            color: ThemeManager.accentColor
                        }
                        radius: width / 2
                        antialiasing: true
                        opacity: sendButton.isActiveItem ? 1.0 : 0
                    }
                }

                contentItem: Item {
                    anchors.fill: parent

                    Rectangle {
                        visible: sendButton.enabled && (sendButton.hovered || sendButton.pressed)
                        anchors.fill: parent
                        radius: width / 2
                        color: ThemeManager.buttonColor
                        opacity: 0.15
                        antialiasing: true
                    }

                    OptimizedImage {
                        source: ThemeManager.darkMode ? "../images/icons/dark/arrow_right.svg" : "../images/icons/arrow_right.svg"
                        sourceSize.width: 22
                        sourceSize.height: 22
                        width: 22
                        height: 22
                        anchors.centerIn: parent
                        fillMode: Image.PreserveAspectFit
                        opacity: parent.parent.enabled ? (sendButton.hovered ? 1 : 0.7) : 0.3
                        fadeInDuration: 0
                        showPlaceholder: false
                    }
                }
            }
        }

    }

    Behavior on height {
        NumberAnimation {
            duration: 100
            easing.type: Easing.OutQuad
        }

    }

}
