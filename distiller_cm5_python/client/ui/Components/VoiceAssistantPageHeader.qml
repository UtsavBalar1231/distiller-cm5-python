import QtQuick 2.15
import QtQuick.Controls 2.15
import QtQuick.Layouts 1.15

Rectangle {
    id: header

    property string serverName: "NO SERVER"
    property string statusText: "Ready"
    property bool isConnected: false
    property bool showStatusText: false
    property bool wifiConnected: false
    property string ipAddress: ""
    property string wifiName: ""
    property alias serverSelectButton: serverSelectBtn
    // Keep the alias but point to a dummy item to prevent runtime errors
    property alias darkModeButton: dummyDarkModeBtn
    property alias closeButton: closeBtn
    property alias statusButton: statusBtn
    
    // System stats properties
    property bool showSystemStats: bridge && bridge.ready ? bridge.getShowSystemStats() : true
    property var systemStats: {"cpu": "N/A", "ram": "N/A", "temp": "N/A", "llm": "Local"}

    // Update WiFi status from bridge
    function updateWifiStatus() {
        if (bridge && bridge.ready) {
            var ipAddr = bridge.getWifiIpAddress();
            wifiConnected = ipAddr && ipAddr !== "No network IP found" && !ipAddr.includes("Error");
            ipAddress = wifiConnected ? ipAddr : "";
            
            // Get WiFi name if available from the bridge
            if (wifiConnected && bridge.getWifiName) {
                wifiName = bridge.getWifiName();
            } else {
                wifiName = "";
            }
        } else {
            wifiConnected = false;
            ipAddress = "";
            wifiName = "";
        }
    }
    
    // Update system stats from bridge
    function updateSystemStats() {
        if (bridge && bridge.ready && showSystemStats) {
            systemStats = bridge.getSystemStats();
            // Also update WiFi status when updating stats
            updateWifiStatus();
        }
    }

    signal serverSelectClicked
    // Keep the signal to prevent errors
    signal darkModeClicked
    signal closeAppClicked
    signal showToastMessage(string message, int duration)

    color: ThemeManager.backgroundColor
    Component.onCompleted: {
        updateWifiStatus();
        updateSystemStats();
    }

    // Dummy invisible item to satisfy the darkModeButton alias
    Item {
        id: dummyDarkModeBtn
        visible: false
        property bool navigable: false
    }

    // Shadow effect for the header
    Rectangle {
        anchors.top: parent.bottom
        anchors.left: parent.left
        anchors.right: parent.right
        height: 1
        color: ThemeManager.black 
    }

    // Server selection button - centered with reduced width
    AppButton {
        id: serverSelectBtn
        width: 120
        height: ThemeManager.buttonHeight
        anchors.left: parent.left
        anchors.leftMargin: ThemeManager.spacingSmall
        anchors.verticalCenter: parent.verticalCenter
        navigable: true
        isFlat: false
        text: "" // Set empty text since we're using custom content
        onClicked: {
            if (statsPopup.visible) {
                statsPopup.close();
            }
            header.serverSelectClicked()
        }
        
        // Custom content using a child Column instead of contentItem
        Column {
            anchors.fill: parent
            anchors.leftMargin: ThemeManager.spacingSmall
            anchors.rightMargin: ThemeManager.spacingSmall
            anchors.topMargin: ThemeManager.spacingTiny
            anchors.bottomMargin: ThemeManager.spacingTiny
            spacing: ThemeManager.spacingTiny / 4

            // Server name - left aligned
            Text {
                width: parent.width
                height: parent.height / 2
                horizontalAlignment: Text.AlignLeft
                text: isConnected && serverName && serverName !== "NO SERVER" ? serverName : "SELECT SERVER"
                font: FontManager.small
                color: serverSelectBtn.visualFocus ? ThemeManager.backgroundColor : ThemeManager.textColor
                elide: Text.ElideRight
            }

            // Status text - left aligned, smaller font
            Text {
                width: parent.width
                height: parent.height / 2
                horizontalAlignment: Text.AlignLeft
                text: statusText
                // visible: isConnected && serverName && serverName !== "NO SERVER"
                font: FontManager.tiny
                color: serverSelectBtn.visualFocus ? ThemeManager.backgroundColor : ThemeManager.textColor
                elide: Text.ElideRight
            }
        }
    }

    // System Status button 
    AppButton {
        id: statusBtn
        width: ThemeManager.buttonHeight
        height: ThemeManager.buttonHeight
        anchors.right: closeBtn.left
        anchors.rightMargin: ThemeManager.spacingSmall
        anchors.verticalCenter: parent.verticalCenter
        navigable: true
        isFlat: true
        buttonRadius: width / 2
        onClicked: {
            if (statsPopup.visible) {
                statsPopup.close();
                statusBtn.forceActiveFocus();
            } else {
                updateSystemStats();
                statsPopup.open();
            }
        }
        
        // Status button icon
        Rectangle {
            parent: statusBtn
            anchors.fill: parent
            color: ThemeManager.backgroundColor
            
            // High contrast highlight for e-ink when focused
            Rectangle {
                visible: statusBtn.visualFocus || statusBtn.pressed || true
                anchors.fill: parent
                radius: width / 2
                color: statusBtn.visualFocus ? ThemeManager.textColor : ThemeManager.backgroundColor
                border.width: ThemeManager.borderWidth
                border.color: ThemeManager.black
                antialiasing: true
            }
            
            Text {
                text: "" 
                font.pixelSize: parent.width * 0.5
                font.family: FontManager.primaryFontFamily
                color: statusBtn.visualFocus ? ThemeManager.backgroundColor : ThemeManager.textColor
                anchors.centerIn: parent
            }
        }
    }
    
    // System stats popup
    Popup {
        id: statsPopup
        x: Math.max(0, parent.width / 2 - width / 2)  // Center horizontally
        y: header.height
        width: parent.width 
        height: parent.parent.height * 0.7  // Cover most of the conversation area height
        padding: ThemeManager.spacingSmall
        closePolicy: Popup.CloseOnEscape | Popup.CloseOnPressOutside
        modal: true
        focus: true
        
        onClosed: {
            // Return focus to the status button when popup closes
            statusBtn.forceActiveFocus();
        }
        
        background: Rectangle {
            color: ThemeManager.backgroundColor
        }
        
        contentItem: Column {
            spacing: ThemeManager.spacingSmall
            width: parent.width
            
            // Header with close button
            Rectangle {
                width: parent.width
                height: ThemeManager.buttonHeight
                color: ThemeManager.transparentColor
                
                // Title
                Text {
                    anchors.left: parent.left
                    anchors.verticalCenter: parent.verticalCenter
                    text: "System Status"
                    font: FontManager.mediumBold
                    color: ThemeManager.textColor
                }
            }
            
            Rectangle {
                width: parent.width
                height: 1
                color: ThemeManager.backgroundColor
            }
            
            // Scrollable content for the stats
            ScrollView {
                width: parent.width
                height: statsPopup.height - 80
                clip: true
                ScrollBar.vertical.policy: ScrollBar.AsNeeded
                
                // Stats content
                GridLayout {
                    id: gridLayout
                    width: parent.width
                    columns: 2
                    rowSpacing: ThemeManager.spacingNormal
                    columnSpacing: ThemeManager.spacingLarge
                    focus: true
                    
                    // CPU usage
                    Text {
                        text: "CPU:"
                        font.pixelSize: FontManager.fontSizeSmall
                        font.family: FontManager.primaryFontFamily
                        color: ThemeManager.textColor
                        Layout.alignment: Qt.AlignLeft
                    }
                    
                    Text {
                        text: systemStats.cpu
                        font.pixelSize: FontManager.fontSizeSmall
                        font.family: FontManager.primaryFontFamily
                        color: ThemeManager.textColor
                        Layout.alignment: Qt.AlignRight
                        Layout.fillWidth: true
                    }
                    
                    // RAM usage
                    Text {
                        text: "Memory:"
                        font.pixelSize: FontManager.fontSizeSmall
                        font.family: FontManager.primaryFontFamily
                        color: ThemeManager.textColor
                        Layout.alignment: Qt.AlignLeft
                    }
                    
                    Text {
                        text: systemStats.ram
                        font.pixelSize: FontManager.fontSizeSmall
                        font.family: FontManager.primaryFontFamily
                        color: ThemeManager.textColor
                        Layout.alignment: Qt.AlignRight
                        Layout.fillWidth: true
                    }
                    
                    // Temperature
                    Text {
                        text: "Temperature:"
                        font.pixelSize: FontManager.fontSizeSmall
                        font.family: FontManager.primaryFontFamily
                        color: ThemeManager.textColor
                        Layout.alignment: Qt.AlignLeft
                    }
                    
                    Text {
                        text: systemStats.temp
                        font.pixelSize: FontManager.fontSizeSmall
                        font.family: FontManager.primaryFontFamily
                        color: ThemeManager.textColor
                        Layout.alignment: Qt.AlignRight
                        Layout.fillWidth: true
                    }
                    
                    // LLM model
                    Text {
                        text: "LLM:"
                        font.pixelSize: FontManager.fontSizeSmall
                        font.family: FontManager.primaryFontFamily
                        color: ThemeManager.textColor
                        Layout.alignment: Qt.AlignLeft
                    }
                    
                    Text {
                        text: systemStats.llm
                        font.pixelSize: FontManager.fontSizeSmall
                        font.family: FontManager.primaryFontFamily
                        color: ThemeManager.textColor
                        Layout.alignment: Qt.AlignRight
                        Layout.fillWidth: true
                    }
                    
                    // WiFi status
                    Text {
                        text: "WiFi:"
                        font.pixelSize: FontManager.fontSizeSmall
                        font.family: FontManager.primaryFontFamily
                        color: ThemeManager.textColor
                        Layout.alignment: Qt.AlignLeft
                    }
                    
                    Text {
                        text: wifiConnected ? (wifiName !== "" ? wifiName : "Connected") : "Disconnected"
                        font.pixelSize: FontManager.fontSizeSmall
                        font.family: FontManager.primaryFontFamily
                        color: ThemeManager.textColor
                        Layout.alignment: Qt.AlignRight
                        Layout.fillWidth: true
                    }
                    
                    // IP Address (only shown if WiFi is connected)
                    Text {
                        text: "IP Address:"
                        font.pixelSize: FontManager.fontSizeSmall
                        font.family: FontManager.primaryFontFamily
                        color: ThemeManager.textColor
                        Layout.alignment: Qt.AlignLeft
                        visible: wifiConnected && ipAddress !== ""
                    }
                    
                    Text {
                        text: ipAddress
                        font.pixelSize: FontManager.fontSizeSmall
                        font.family: FontManager.primaryFontFamily
                        color: ThemeManager.textColor
                        Layout.alignment: Qt.AlignRight
                        Layout.fillWidth: true
                        visible: wifiConnected && ipAddress !== ""
                    }
                }
            }
        }
    }

    // Close app button - positioned at top right
    AppButton {
        id: closeBtn
        width: ThemeManager.buttonHeight
        height: ThemeManager.buttonHeight
        anchors.right: parent.right
        anchors.rightMargin: ThemeManager.spacingSmall
        anchors.verticalCenter: parent.verticalCenter
        navigable: true
        isFlat: true
        buttonRadius: width / 2
        onClicked: {
            if (statsPopup.visible) {
                statsPopup.close();
            }
            shutdownConfirmDialog.open();
        }
        
        // Shutdown button icon
        Rectangle {
            parent: closeBtn
            anchors.fill: parent
            color: ThemeManager.backgroundColor
            
            // High contrast highlight for e-ink when focused
            Rectangle {
                visible: closeBtn.visualFocus || closeBtn.pressed || true
                anchors.fill: parent
                radius: width / 2
                color: closeBtn.visualFocus ? ThemeManager.textColor : ThemeManager.backgroundColor
                border.width: ThemeManager.borderWidth
                border.color: ThemeManager.black
                antialiasing: true
            }
            
            Text {
                text: "" // Power/Shutdown icon
                font.pixelSize: parent.width * 0.3
                font.family: FontManager.primaryFontFamily
                color: closeBtn.visualFocus ? ThemeManager.backgroundColor : ThemeManager.textColor
                anchors.centerIn: parent
            }
        }
    }
    
    // Close app timer - simple delay before closing
    Timer {
        id: closeAppTimer
        interval: 2000  // 2 second delay before closing
        repeat: false
        running: false
        
        onTriggered: {
            if (bridge && bridge.ready) {
                // Signal app shutdown via UART
                header.closeAppClicked();
                // Then execute system shutdown
                bridge.executeSystemCommand("shutdown now");
            }
        }
    }
    
    // Shutdown confirmation dialog
    AppDialog {
        id: shutdownConfirmDialog
        dialogTitle: "System Shutdown"
        message: "Are you sure you want to shut down the system?"
        standardButtonTypes: DialogButtonBox.Yes | DialogButtonBox.No
        yesButtonText: "Proceed"
        noButtonText: "Cancel"
        acceptButtonColor: ThemeManager.backgroundColor
        
        onAccepted: {
            // Close the dialog
            shutdownConfirmDialog.close();
            
            // Show shutdown message
            header.showToastMessage("Shutting down...", 5000);
            
            // Start the timer to delay closing
            closeAppTimer.start();
        }
        
        onRejected: {
            shutdownConfirmDialog.close();
        }
    }
}
