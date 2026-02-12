import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox
import pyaudio
import numpy as np
import scipy.signal as signal
import threading
import queue
import time
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from collections import deque

# --- Constants ---
LOCKOUT_MS = 80 
LOCKOUT_SAMPLES = int(44100 * (LOCKOUT_MS / 1000.0))
# Standard BPH list (Watch + Clock frequencies)
STANDARD_BPH = [3600, 7200, 14400, 18000, 19800, 21600, 25200, 28800, 36000]
SAMPLE_RATE = 44100
CHUNK_SIZE = 2048
FORMAT = pyaudio.paInt16

class WatchAnalyzer:
    def __init__(self):
        self.p = pyaudio.PyAudio()
        self.stream = None
        self.running = False
        self.data_queue = queue.Queue()
        self.results_queue = queue.Queue()
        
        # AGC Variables
        self.agc_gain = 50.0
        self.target_peak = 20000.0 
        
        # Detection Variables
        self.threshold_percent = 40.0
        self.last_trigger_index = -999999
        self.total_processed_samples = 0
        self.last_tick_time = 0
        
        # Data Storage
        self.intervals = []          # Rolling buffer for "Instant" stats (last 10)
        self.session_intervals = []  # All valid intervals for "Session" stats
        
        # Filter Setup (2kHz - 10kHz)
        self.b, self.a = signal.butter(4, [2000, 10000], btype='bandpass', fs=SAMPLE_RATE)
        self.plot_buffer = np.zeros(SAMPLE_RATE * 2 // 20) 
        
        self.reset_data()

    def reset_data(self):
        self.last_tick_time = 0
        self.intervals = []
        self.session_intervals = []
        self.agc_gain = 50.0
        self.total_processed_samples = 0
        self.last_trigger_index = -999999
        self.results_queue.put(("RESET", None))
        self.results_queue.put(("LOG", "--- Session Reset ---"))

    def get_input_devices(self):
        devices = {}
        for i in range(self.p.get_device_count()):
            try:
                info = self.p.get_device_info_by_index(i)
                if info.get('maxInputChannels') > 0:
                    devices[info.get('name')] = i
            except: pass
        return devices

    def start_stream(self, device_index=None):
        if self.running: return
        self.reset_data()
        self.running = True
        try:
            self.stream = self.p.open(
                format=FORMAT, channels=1, rate=SAMPLE_RATE,
                input=True, input_device_index=device_index,
                frames_per_buffer=CHUNK_SIZE, stream_callback=self._audio_callback
            )
        except Exception as e:
            self.running = False
            self.results_queue.put(("LOG", f"Error opening stream: {e}"))
            return
        
        self.process_thread = threading.Thread(target=self._process_data)
        self.process_thread.daemon = True
        self.process_thread.start()
        self.results_queue.put(("LOG", f"Listening on device {device_index}..."))

    def stop_stream(self):
        self.running = False
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
            self.stream = None

    def _audio_callback(self, in_data, frame_count, time_info, status):
        if self.running:
            self.data_queue.put(np.frombuffer(in_data, dtype=np.int16))
        return (None, pyaudio.paContinue)

    def _process_data(self):
        window_len = int(SAMPLE_RATE * 0.005) 
        smooth_win = np.ones(window_len) / window_len

        while self.running:
            try:
                raw_data = self.data_queue.get(timeout=1)
            except queue.Empty: continue

            # 1. AGC
            chunk_max = np.max(np.abs(raw_data))
            if chunk_max > 0:
                instant_gain = self.target_peak / chunk_max
                self.agc_gain = (self.agc_gain * 0.98) + (instant_gain * 0.02)
                self.agc_gain = max(1.0, min(self.agc_gain, 300.0))

            # 2. Filter & Smooth
            amplified = raw_data.astype(np.float32) * self.agc_gain
            filtered = signal.lfilter(self.b, self.a, amplified)
            envelope = np.abs(filtered)
            smoothed = np.convolve(envelope, smooth_win, mode='same')

            # 3. Peak Detection
            current_threshold = (self.threshold_percent / 100.0) * 32768
            
            for i, sample in enumerate(smoothed):
                global_idx = self.total_processed_samples + i
                
                if global_idx - self.last_trigger_index < LOCKOUT_SAMPLES:
                    continue 
                
                if sample > current_threshold:
                    self.last_trigger_index = global_idx
                    # Notify UI immediately for LED
                    self.results_queue.put(("TICK", None))
                    
                    current_time = global_idx / SAMPLE_RATE
                    
                    if self.last_tick_time > 0:
                        delta = current_time - self.last_tick_time
                        
                        status = "OK"
                        if delta < 0.09: status = "NOISE"
                        elif delta > 2.2: status = "MISSED"
                        
                        self.results_queue.put(("LOG", f"Δ: {delta*1000:.0f}ms -> {status}"))

                        if status == "OK":
                            self.intervals.append(delta)
                            self.session_intervals.append(delta)
                            if len(self.intervals) > 10: self.intervals.pop(0)
                            self._analyze_intervals()
                    
                    self.last_tick_time = current_time

            self.total_processed_samples += len(raw_data)
            
            downsampled = smoothed[::20]
            self.plot_buffer = np.roll(self.plot_buffer, -len(downsampled))
            self.plot_buffer[-len(downsampled):] = downsampled
            
            if self.data_queue.qsize() < 2:
                self.results_queue.put(("WAVEFORM", (self.plot_buffer, current_threshold, self.agc_gain)))

    def _analyze_intervals(self):
        if len(self.session_intervals) < 5: return
        
        # Instant Rate (Median of last 10)
        recent = self.intervals
        avg_instant = np.median(recent) if recent else 1.0
            
        # Session Rate (Linear Regression)
        valid_data = self.session_intervals[2:] # Skip first 2
        if len(valid_data) < 5: return
        
        y_cumulative = np.cumsum(valid_data)
        x_ticks = np.arange(len(y_cumulative))
        
        slope, intercept = np.polyfit(x_ticks, y_cumulative, 1)
        
        measured_bph = 3600 / slope
        target_bph = min(STANDARD_BPH, key=lambda x: abs(x - measured_bph))
        target_interval = 3600 / target_bph
        
        rate_instant = ((target_interval - avg_instant) / target_interval) * 86400
        rate_session = ((target_interval - slope) / target_interval) * 86400
        
        # Beat Error
        be_ms = 0.0
        if len(valid_data) >= 4:
            recent_clean = valid_data[-20:]
            evens, odds = recent_clean[0::2], recent_clean[1::2]
            if len(odds) > 0 and len(evens) > 0:
                be_ms = abs(np.mean(evens) - np.mean(odds)) * 1000
            
        self.results_queue.put(("STATS", {
            "bph": target_bph, 
            "rate_instant": rate_instant, 
            "rate_session": rate_session,
            "be": be_ms,
            "count": len(valid_data)
        }))

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Timegrapher v6 - Final")
        self.geometry("1200x850")
        self.analyzer = WatchAnalyzer()
        self.device_map = self.analyzer.get_input_devices()
        self.tick_timer = None # For LED flash
        self._build_ui()
        self.update_loop()
        self.protocol("WM_DELETE_WINDOW", self._on_close)

    def _build_ui(self):
        main_pane = ttk.PanedWindow(self, orient=tk.HORIZONTAL)
        main_pane.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        left_frame = ttk.Frame(main_pane)
        right_frame = ttk.Frame(main_pane)
        main_pane.add(left_frame, weight=3)
        main_pane.add(right_frame, weight=1)

        # --- Toolbar ---
        toolbar = ttk.Frame(left_frame, padding=5)
        toolbar.pack(fill=tk.X)
        
        ttk.Button(toolbar, text="Start/Stop", command=self.toggle_listen).pack(side=tk.LEFT, padx=5)
        ttk.Button(toolbar, text="Reset", command=self.analyzer.reset_data).pack(side=tk.LEFT, padx=5)
        
        self.device_var = tk.StringVar()
        c = ttk.Combobox(toolbar, textvariable=self.device_var, values=list(self.device_map.keys()), state="readonly", width=25)
        if self.device_map: c.current(0)
        c.pack(side=tk.LEFT, padx=10)

        # LED Canvas
        self.canvas_led = tk.Canvas(toolbar, width=30, height=30, highlightthickness=0)
        self.canvas_led.pack(side=tk.RIGHT, padx=10)
        self.led = self.canvas_led.create_oval(5, 5, 25, 25, fill="#333")
        
        # Audio Checkbox
        self.audio_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(toolbar, text="Click Sound", variable=self.audio_var).pack(side=tk.RIGHT, padx=10)

        # --- Statistics ---
        stats_frame = ttk.LabelFrame(left_frame, text="Measurement", padding=10)
        stats_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Label(stats_frame, text="INSTANT RATE", font=("Arial", 8, "bold"), foreground="#666").grid(row=0, column=0)
        ttk.Label(stats_frame, text="SESSION AVERAGE", font=("Arial", 8, "bold"), foreground="#007acc").grid(row=0, column=1)
        ttk.Label(stats_frame, text="BEAT ERROR", font=("Arial", 8, "bold"), foreground="#666").grid(row=0, column=2)
        
        self.lbl_instant = ttk.Label(stats_frame, text="---", font=("Courier", 24), foreground="#555")
        self.lbl_instant.grid(row=1, column=0, padx=20, pady=5)
        
        self.lbl_session = ttk.Label(stats_frame, text="---", font=("Courier", 32, "bold"), foreground="#007acc")
        self.lbl_session.grid(row=1, column=1, padx=20, pady=5)
        
        self.lbl_be = ttk.Label(stats_frame, text="---", font=("Courier", 24), foreground="#555")
        self.lbl_be.grid(row=1, column=2, padx=20, pady=5)
        
        self.lbl_bph = ttk.Label(stats_frame, text="BPH: ---")
        self.lbl_bph.grid(row=2, column=0)
        
        self.lbl_conf = ttk.Label(stats_frame, text="Confidence: 0 samples")
        self.lbl_conf.grid(row=2, column=1)

        # --- Controls ---
        control_frame = ttk.Frame(left_frame, padding=10)
        control_frame.pack(fill=tk.X)
        
        self.thresh_var = tk.DoubleVar(value=40.0)
        ttk.Label(control_frame, text="Threshold:").pack(side=tk.LEFT)
        ttk.Scale(control_frame, from_=5.0, to=95.0, variable=self.thresh_var, command=self._set_thresh).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=10)
        self.lbl_thresh = ttk.Label(control_frame, text="40%")
        self.lbl_thresh.pack(side=tk.LEFT)
        self.lbl_agc = ttk.Label(control_frame, text="AGC: --", foreground="blue")
        self.lbl_agc.pack(side=tk.RIGHT, padx=10)

        # --- Plot ---
        self.fig = Figure(figsize=(5, 4), dpi=100)
        self.ax = self.fig.add_subplot(111)
        self.ax.set_facecolor('#f5f5f5')
        self.ax.set_ylim(0, 35000)
        self.ax.get_xaxis().set_visible(False)
        self.ax.get_yaxis().set_visible(False)
        
        self.line, = self.ax.plot([], [], color='#228B22', lw=1)
        self.tline, = self.ax.plot([], [], color='red', ls='--', alpha=0.6, lw=1)
        
        self.canvas = FigureCanvasTkAgg(self.fig, left_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # --- Log ---
        log_lbl = ttk.Label(right_frame, text="Real-time Log", font=("Arial", 10, "bold"))
        log_lbl.pack(anchor=tk.W, pady=5)
        self.log_box = scrolledtext.ScrolledText(right_frame, width=25, height=40, state='disabled', font=("Consolas", 9))
        self.log_box.pack(fill=tk.BOTH, expand=True)
        self.log_box.tag_config("OK", foreground="green")
        self.log_box.tag_config("NOISE", foreground="red")
        self.log_box.tag_config("MISSED", foreground="orange")
        self.log_box.tag_config("INFO", foreground="black")

    def _set_thresh(self, v):
        self.analyzer.threshold_percent = float(v)
        self.lbl_thresh.config(text=f"{int(float(v))}%")

    def toggle_listen(self):
        if not self.analyzer.running:
            idx = self.device_map.get(self.device_var.get())
            self.analyzer.start_stream(idx)
        else:
            self.analyzer.stop_stream()

    def log_msg(self, msg):
        self.log_box.config(state='normal')
        tag = "INFO"
        if "OK" in msg: tag = "OK"
        elif "NOISE" in msg: tag = "NOISE"
        elif "MISSED" in msg: tag = "MISSED"
        self.log_box.insert(tk.END, msg + "\n", tag)
        self.log_box.see(tk.END)
        self.log_box.config(state='disabled')

    def update_loop(self):
        try:
            while True:
                tag, data = self.analyzer.results_queue.get_nowait()
                
                if tag == "TICK":
                    # Flash LED
                    self.canvas_led.itemconfig(self.led, fill="#00FF00")
                    if self.tick_timer: self.after_cancel(self.tick_timer)
                    self.tick_timer = self.after(50, lambda: self.canvas_led.itemconfig(self.led, fill="#333"))
                    # Play Sound (if checked)
                    if self.audio_var.get():
                        self.bell() # System beep
                
                elif tag == "WAVEFORM":
                    buf, thresh, agc = data
                    self.line.set_data(np.arange(len(buf)), buf)
                    self.tline.set_data([0, len(buf)], [thresh, thresh])
                    self.ax.set_xlim(0, len(buf))
                    self.lbl_agc.config(text=f"AGC: {agc:.1f}x")
                    self.canvas.draw_idle()
                    
                elif tag == "LOG":
                    self.log_msg(data)
                    
                elif tag == "RESET":
                    self.lbl_instant.config(text="---")
                    self.lbl_session.config(text="---")
                    
                elif tag == "STATS":
                    self.lbl_instant.config(text=f"{data['rate_instant']:+.0f} s/d")
                    self.lbl_session.config(text=f"{data['rate_session']:+.1f} s/d")
                    self.lbl_be.config(text=f"{data['be']:.1f} ms")
                    self.lbl_bph.config(text=f"BPH: {data['bph']}")
                    
                    n = data['count']
                    conf_text = f"Confidence: {n} samples"
                    if n < 10: conf_text += " (Low)"
                    elif n > 60: conf_text += " (High)"
                    self.lbl_conf.config(text=conf_text)
                    
        except queue.Empty: pass
        self.after(50, self.update_loop)

    def _on_close(self):
        self.analyzer.stop_stream()
        self.destroy()

if __name__ == "__main__":
    App().mainloop()
