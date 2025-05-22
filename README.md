# Kiosk Accessibility Enhancement Project

[한국어](./README_ko.md)

## Introduction

This project aims to significantly improve the accessibility of digital kiosks for visually impaired individuals. By leveraging Optical Character Recognition (OCR) and a web-based interface, users can interact with kiosk systems using their own smartphones, overcoming many of the traditional barriers faced by those with visual impairments.

## How it Works

The system operates in a sequence of steps to provide an accessible interface:

1.  **Screen Capture:** The kiosk's screen content is captured.
2.  **OCR Processing:** The captured image is processed by an OCR engine to extract all readable text.
3.  **Web Interface Generation:** The extracted text and its approximate layout are used to generate a simplified web page. This page is accessible on the user's smartphone or other web-enabled devices.
4.  **User Interaction:** The user interacts with the web page (e.g., by tapping on a button representing a kiosk option).
5.  **Coordinate Mapping & Mouse Emulation:** The system identifies the coordinates of the selected text element on the original kiosk screen. It then emulates a mouse click at these coordinates, effectively transmitting the user's choice to the kiosk system.

## Features

*   **Remote Kiosk Control:** Interact with kiosks using a familiar smartphone interface.
*   **Text-to-Speech Friendly:** The web interface is designed to be compatible with screen readers.
*   **Multi-Language Support:** Initial OCR capabilities include support for Korean. (This can be expanded).
*   **Non-Invasive:** Works as an overlay without requiring modification to the underlying kiosk software.

## Usage Example

Imagine a user approaching a self-service kiosk. Instead of struggling to read the screen or locate touch targets, they can:

1.  Activate our system (e.g., by scanning a QR code or connecting to a local Wi-Fi network provided by the system).
2.  Open the web interface on their smartphone.
3.  See the kiosk's options presented as clearly labeled buttons or text elements.
4.  Tap a button on their phone.
5.  The system then clicks the corresponding button on the kiosk.

**Visual Demonstration:**

To better understand the flow, please see the following examples (actual files to be added by developers in the `assets` directory):

*   **Usage Flow GIF:** `![User Interaction Flow](./assets/kiosk_usage_flow.gif)` (Illustrates the complete interaction from phone to kiosk)
*   **OCR Interface Example:** `![OCR Web Interface](./assets/ocr_interface_example.png)` (Shows an example of the kiosk screen rendered as a web page)

*(Note: Developers, please create an `assets` directory in the project root and place `kiosk_usage_flow.gif` and `ocr_interface_example.png` inside it.)*

## How to Use (End User)

1.  **Connect to the System:** Follow the specific instructions provided at the kiosk location to connect your smartphone to the accessibility system (this might involve scanning a QR code, connecting to a specific Wi-Fi network, or opening a specific URL in your phone's browser).
2.  **Navigate the Web Interface:** Once connected, the kiosk's screen content will be displayed in your web browser. Use your phone's accessibility features (like screen readers) if needed.
3.  **Make Selections:** Tap on the buttons or links shown on the web page that correspond to the options you want to select on the kiosk.
4.  **Confirmation:** The kiosk should respond as if you had touched its screen directly.

## For Developers

This project utilizes the following key technologies:

*   **Python:** For the core application logic.
*   **OCR Engine (Tesseract):** For text extraction from screen captures.
*   **Web Server (e.g., Flask/custom):** To serve the accessible web interface.
*   **Mouse Emulation Library:** To interact with the kiosk system.

**Basic Setup (Conceptual):**

1.  Ensure Python and Tesseract OCR (with Korean language data) are installed.
2.  Clone this repository.
3.  Install dependencies (typically from a `requirements.txt` file - not yet present, but would be added in a full project setup).
4.  Run the main application script (e.g., `src/main.py`).

Further details on setting up a development environment and contributing will be added in the future.

---

*This README provides a general overview. Specific implementation details can be found by exploring the `src` directory.*
