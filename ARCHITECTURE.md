# AI Automotive System – Compact Technical Architecture

The system is designed as a modular AI-driven automotive platform composed of three logical layers: an automotive microcontroller layer (vehicle interface), an edge compute layer (voice and audio processing), and a central AI server layer (intelligence and orchestration). The architecture is non-invasive, secure by design, and fully independent from Android Auto protocols.

At the vehicle level, the core controller is an Arduino UNO R4 WiFi. This unit is responsible for real-time telemetry acquisition, GPS integration, LED ambient control, and secure communication with the AI server. Vehicle data is collected through an OBD2 interface using a CAN controller (e.g., MCP2515 over SPI). Standard PID polling retrieves RPM, vehicle speed, coolant temperature, engine load, throttle position, fuel level, and DTC codes. The system operates in read-only mode by default. Any write-capable CAN interaction must pass through a strict command whitelist and validation logic.

GPS functionality is implemented using a u-blox NEO-M8N module connected via UART. Parsed data includes latitude, longitude, altitude, speed over ground, timestamp, and HDOP precision metrics. GPS and telemetry data are serialized into structured JSON packets and transmitted securely (HTTPS or WSS) to the AI backend.

An integrated LED control subsystem is managed directly by the Arduino. Addressable LEDs (WS2812B or SK6812 RGBW) are driven using non-blocking control logic (FastLED library). Power is supplied through a dedicated automotive-rated step-down converter with isolation and filtering. LED behavior is governed by AI-issued structured commands, supporting static ambient mode, RPM-reactive mode, speed-adaptive effects, music-reactive visualization, anomaly alert signaling, and AI mood-based lighting states. All LED commands are validated against vehicle state conditions before execution.

The edge compute layer is implemented on a Raspberry Pi 5 (recommended). This node manages wake word detection, speech-to-text (Whisper or Vosk), audio routing, TTS playback (Piper), and communication bridging between Arduino and the AI server. It also handles local fallback logic if server connectivity is lost. The voice pipeline follows: Wake Word → STT → AI Request → AI Response → TTS → Audio Injection.

Android Auto is not modified or intercepted. Audio integration is achieved through hardware-level strategies. Recommended architecture: phone outputs Android Auto to head unit normally; Raspberry outputs TTS through a USB DAC into AUX or a dedicated mixer. Ducking is implemented via one of three methods: hardware MUTE line triggering, CAN-based mute frame replication (after reverse engineering), or analog mixer with sidechain compression. When AI speech is triggered, Android Auto volume is reduced, TTS is injected, then original volume is restored.

The AI server layer runs on a GPU-enabled local machine or cloud instance. It exposes secure REST/WebSocket APIs for telemetry ingestion, voice processing, LED control logic, diagnostics interpretation, and music orchestration. Telemetry is stored in a time-series database for driving pattern analysis and predictive modeling. The AI layer performs natural language reasoning, anomaly detection, driving style classification, adaptive music selection, and contextual lighting generation.

Music integration is preferably handled through a personal music server (e.g., Jellyfin, Navidrome, or custom Node backend). This allows full control without DRM restrictions. The AI can dynamically generate playlists based on speed, RPM, time of day, GPS location, or detected driving mood. Audio normalization, crossfading, and adaptive transitions are handled at the edge compute layer. Optional integration with external streaming services is possible through official APIs only.

Networking is achieved through either Bluetooth PAN tethering, a dedicated LTE/5G modem attached to the Raspberry, or a fully offline configuration using local AI inference. The recommended configuration for reliability is a dedicated mobile data module to decouple system availability from the phone.

Security is enforced through strict architectural separation. The AI layer never sends raw CAN frames directly. All actuation commands pass through a validation layer that translates high-level intents into pre-approved tokens executable by the Arduino. Safeguards include command whitelisting, packet signing, rate limiting, watchdog timers, and automatic safe mode activation when vehicle speed exceeds defined thresholds. Safety-critical systems remain completely untouched.

Implementation progresses in four phases: passive telemetry acquisition and dashboard logging; voice interaction and AI conversational capability; intelligent LED and adaptive music control; advanced analytics including predictive maintenance and behavioral modeling.

The final system is a distributed, AI-assisted automotive platform featuring real-time telemetry awareness, adaptive ambient lighting, intelligent audio orchestration, secure command isolation, and full compatibility with Android Auto without protocol interference.

---

# Required Hardware and Materials

## Core Controllers

* Arduino UNO R4 WiFi (ABX00087)
* Raspberry Pi 5 (recommended, 4GB or 8GB)

## Vehicle Interface Components

* MCP2515 CAN bus module (SPI) or automotive-grade CAN shield
* OBD2 connector cable (male)
* Automotive-rated DC-DC step-down converter (12V → 5V, isolated preferred)
* Inline fuse (automotive blade type)
* TVS diode for surge protection (automotive transient suppression)

## GPS Module

* u-blox NEO-M8N GNSS module
* External active GPS antenna (SMA)

## LED System

* WS2812B or SK6812 RGB/RGBW LED strip
* Logic level shifter (if required for signal stability)
* Dedicated 5V high-current power supply rail
* Inline fuse for LED power line

## Audio System

* USB DAC (for Raspberry audio output)
* 3.5mm AUX cable or RCA interface
* Optional analog audio mixer with sidechain compression
* Optional automotive relay (for MUTE line control)

## Networking

* Bluetooth PAN (no extra hardware required) OR
* USB LTE/5G modem + SIM card
* External antenna for cellular modem (recommended)

## Server Infrastructure

* Local GPU server (RTX-class recommended) OR
* Dedicated mini-server / VPS instance
* Time-series database (e.g., InfluxDB)
* Reverse proxy (e.g., Nginx)
* SSL certificate (Let’s Encrypt or equivalent)

## Software Stack

* Arduino firmware (C++ with CAN + FastLED + JSON serialization)
* Raspberry OS (64-bit)
* Whisper or Vosk (STT)
* Piper (TTS)
* Node.js or Python backend API
* Personal music server (Jellyfin / Navidrome / custom backend)

## Optional Enhancements

* OLED diagnostic display (I2C)
* Rotary encoder or capacitive touch input
* IMU sensor (MPU6050) for motion analysis
* Microphone array (for noise reduction)
* Active cooling solution for Raspberry
