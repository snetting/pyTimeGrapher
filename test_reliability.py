import queue
import threading
import unittest
from types import SimpleNamespace

import numpy as np
import scipy.signal as signal


class ReliabilityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import pyTimeGrapher
        cls.mod = pyTimeGrapher

    def _analyzer(self):
        analyzer = self.mod.WatchAnalyzer.__new__(self.mod.WatchAnalyzer)
        analyzer.b, analyzer.a = signal.butter(4, [2000, 10000], btype='bandpass', fs=self.mod.SAMPLE_RATE)
        analyzer._filter_zi = signal.lfilter_zi(analyzer.b, analyzer.a) * 0.0
        analyzer._dc_offset = None
        analyzer._envelope_history = np.zeros(int(self.mod.SAMPLE_RATE * 0.005) - 1, dtype=np.float32)
        analyzer.agc_gain = 1.0
        return analyzer

    def test_chunked_dsp_matches_continuous_processing(self):
        raw = (350 + 100 * np.sin(2 * np.pi * 4000 * np.arange(4096) / self.mod.SAMPLE_RATE)).astype(np.float32)
        window = np.ones(int(self.mod.SAMPLE_RATE * 0.005)) / (self.mod.SAMPLE_RATE * 0.005)
        continuous_analyzer = self._analyzer()
        continuous = continuous_analyzer._smooth_envelope(
            np.abs(continuous_analyzer._filter_chunk(raw)), window)
        chunked_analyzer = self._analyzer()
        chunks = []
        for chunk in np.split(raw, 2):
            chunks.append(chunked_analyzer._smooth_envelope(
                np.abs(chunked_analyzer._filter_chunk(chunk)), window))
        np.testing.assert_allclose(np.concatenate(chunks), continuous, atol=3e-4)

    def test_stop_stream_waits_for_worker_and_drains_audio(self):
        analyzer = SimpleNamespace(running=True, stream=None, process_thread=None,
                                   data_queue=queue.Queue(), _stop_event=threading.Event())
        started = threading.Event()

        def worker():
            started.set()
            while not analyzer._stop_event.is_set():
                try:
                    analyzer.data_queue.get(timeout=0.1)
                except queue.Empty:
                    pass

        analyzer.process_thread = threading.Thread(target=worker)
        analyzer.process_thread.start()
        analyzer.data_queue.put(np.ones(4))
        started.wait(1)
        self.mod.WatchAnalyzer.stop_stream(analyzer)
        self.assertFalse(analyzer.process_thread)
        self.assertTrue(analyzer.data_queue.empty())

    def test_cadence_gap_is_not_classified_as_a_valid_interval(self):
        analyzer = SimpleNamespace(cadence_interval=0.125)
        classify = self.mod.WatchAnalyzer._classify_interval
        self.assertEqual(classify(analyzer, 0.125), "OK")
        self.assertEqual(classify(analyzer, 0.25), "MISSED")
        self.assertEqual(classify(analyzer, 0.05), "NOISE")


if __name__ == '__main__':
    unittest.main()
