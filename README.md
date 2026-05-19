# OCR Number Classifier — ESP32 + TensorFlow Lite

A hand-drawn digit recognition system. Draw a digit on a Python GUI, send it to an ESP32 microcontroller over USB serial, and get a real-time prediction back — with live correction support to improve accuracy on the fly.

---

## Features

- On-device inference running entirely on the ESP32 (no cloud, no PC compute)
- Python drawing GUI with COM port selector
- Real-time prediction displayed in the GUI after each draw
- Online weight correction — send the right label when the prediction is wrong and the ESP32 updates its weights immediately

---

## Requirements

**Hardware**
- Espressif ESP32-DevKitC v4 (or ESP32-WROOM-DA)
- USB cable to connect ESP32 to your PC

**Software**
```bash
pip install tensorflow scikit-learn matplotlib pyserial
```
- Arduino IDE 2.x
- ESP32 board support + TensorFlow Lite library installed in Arduino IDE

---

## Setup Roadmap

### Step 1 — Install ESP32 board support in Arduino IDE

1. Open Arduino IDE → **File → Preferences**
2. In *Additional Board Manager URLs*, paste:
```
   https://raw.githubusercontent.com/espressif/arduino-esp32/gh-pages/package_esp32_index.json
```
3. Go to **Tools → Board → Boards Manager**, search `esp32`, install **esp32 by Espressif Systems**
4. Go to **Tools → Board** and select **ESP32 Dev Module**

### Step 2 — Install TensorFlow Lite library

In Arduino IDE → **Sketch → Include Library → Manage Libraries**
Search for `TensorFlowLite_ESP32` and install it.

### Step 3 — Train the model (skip if already done)

```bash
jupyter notebook src/plot_digits_last_image.ipynb
```

Run all cells. This saves `build/digit_classifier.tflite`.

### Step 4 — Convert model to C array

```bash
xxd -i build/digit_classifier.tflite > src/mysketch/digit_classifier.h
```

### Step 5 — Flash the ESP32

1. Open `src/mysketch/mysketch.ino` in Arduino IDE
2. Go to **Tools → Port** and select your ESP32's port
   - Windows: e.g. `COM14`
   - Linux: e.g. `/dev/ttyUSB0`
3. Click the **Upload (→)** button
4. Wait for:
```
   Hash of data verified.
   Hard resetting via RTS pin...
```

### Step 6 — Verify it's running

Open Serial Monitor (`Ctrl + Shift + M`), set baud rate to **115200**.
If the screen is blank, press the **EN/RST** button on the ESP32 once. You should see:

```
ESP32 Perceptron Classifier ready.
```

> ⚠️ Close the Serial Monitor before running the Python GUI — both cannot use the same port at the same time.

### Step 7 — Run the Python GUI

```bash
cd src
python drawing.py
```

---

## How to Use the Classifier

1. Select your COM port from the dropdown (e.g. `COM14`)
2. Draw a digit (0–9) on the grid using your mouse — draw **small and centered**, not edge-to-edge
3. Click **Send to ESP32**
4. The predicted digit appears below the grid

---

## How to Use the Correction Feature

If the prediction is wrong:

1. Type the correct digit in the **Correct label (0–9)** box
2. Click **Send Correction**
3. The ESP32 responds with `OK:<label>` and updates its weights
4. Draw the same digit again — it will progressively improve

> Only send a correction when the prediction is actually wrong. Sending a wrong label will confuse the model.
