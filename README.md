# pyTimegrapher ⏱️

**pyTimegrapher** is a Python-based acoustic analysis tool designed to act as a software timegrapher for regulating mechanical watches and pendulum clocks. It serves as a free, open-source alternative to dedicated hardware (like the Weishi No. 1000 or Witschi Watch Expert), using your computer's microphone or a contact piezo sensor to measure the heartbeat of a movement.

## 🚀 Key Features

* **Universal Compatibility:** Works with both fast-beat wristwatches (14,400 to 36,000 bph) and slow-beat pendulum clocks (3,600 to 7,200 bph).
* **Dual-Mode Analysis:**
    * **Instant Rate:** Shows real-time stability and immediate escapement noise.
    * **Session Average:** Uses **Linear Regression** (slope detection) to provide a "Chronometer-grade" daily rate (s/d) that ignores short-term jitter.
* **Smart Noise Filtering:**
    * **Dead Time Lockout:** Automatically ignores "ringing" and echoes (80ms lockout) to prevent double-triggering on loud movements.
    * **Auto-Gain Control (AGC):** Dynamically adjusts microphone sensitivity to keep the signal lock steady.
* **Beat Error Measurement:** Calculates the milliseconds of error between the "tick" and the "tock" (pallet fork centering).
* **Visual Feedback:** Real-time waveform scope, simulated LED beat indicator, and optional audio click.

## 🛠️ Installation

### Prerequisites

You will need Python 3.x and the following libraries:

```bash
pip install numpy scipy matplotlib pyaudio
```

*Note for Linux Users:* You may need to install the PortAudio development headers before installing `pyaudio`:
```bash
sudo apt-get install python3-pyaudio portaudio19-dev
```

### Hardware Setup 🎙️

While a standard microphone works for loud clocks, **mechanical wristwatches require a contact microphone** for accurate results due to their low acoustic energy.

1.  **Best:** A generic "Clip-on Guitar Pickup" or Piezo disc taped to the watch case.
2.  **Good:** A high-quality lapel mic clipped directly to the crown.
3.  **Okay:** A standard webcam mic (works only for loud pocket watches or pendulum clocks).

## 📖 How to Use

1.  **Launch the App:**
    ```bash
    python timegrapher.py
    ```
2.  **Select Input:** Choose your microphone from the dropdown list.
3.  **Start Listening:** Click **Start/Stop**.
4.  **Calibration:**
    * Watch the **Waveform** graph. You want clear, distinct spikes.
    * Adjust the **Threshold** slider until the red line sits just above the background noise but below the main spikes.
    * The **LED** in the top right should flash steadily with every tick.
5.  **Read the Stats:**
    * Wait about 30-60 seconds for the **Session Average** to stabilize.
    * **Rate:** How many seconds per day the watch is gaining (+) or losing (-).
    * **Beat Error (B.E.):** The timing difference between the "tick" and "tock." A reading of 0.0ms is perfect; anything under 2.0ms is acceptable.

## 🧮 How It Works

OpenTimegrapher uses a digital signal processing (DSP) pipeline:

1.  **Bandpass Filter:** Isolates frequencies between 2kHz and 10kHz (the "snap" of the pallet fork).
2.  **Envelope & Smoothing:** Converts the raw audio spikes into a smooth "hill" for detection.
3.  **Heuristic Lockout:** Once a tick is detected, the sensor goes "blind" for 80ms to ignore reverberation.
4.  **Linear Regression:** Instead of averaging simple intervals (which may drift wildly), the Session Rate calculates the *slope* of the total elapsed time, offering mathematically superior stability for long-term regulation.

## 🤝 Contributing

Pull requests are welcome! We are currently looking for:

* Algorithm improvements for low-amplitude movements.
* Lift Angle / Amplitude calculation support.


