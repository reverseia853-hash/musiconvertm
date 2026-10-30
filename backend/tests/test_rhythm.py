import unittest

import numpy as np

import rhythm


class TempoAnalysisTests(unittest.TestCase):
    @staticmethod
    def click_track(bpm: int, duration: int = 30) -> np.ndarray:
        sample_rate = rhythm.SR
        audio = np.zeros(sample_rate * duration, dtype=np.float32)
        period = 60 / bpm
        click = np.exp(-np.arange(500) / 65).astype(np.float32)
        for beat_index, time in enumerate(np.arange(0, duration, period)):
            start = int(time * sample_rate)
            end = min(start + len(click), len(audio))
            strength = 1.0 if beat_index % 4 == 0 else 0.5
            audio[start:end] += strength * click[:end - start]
        return audio

    def test_estimates_common_synthetic_tempos_within_eight_percent(self) -> None:
        for expected in (30, 40, 50, 60, 90, 120, 140, 160):
            with self.subTest(expected=expected):
                result = rhythm.analyze(self.click_track(expected))
                self.assertLessEqual(abs(result["bpm"] - expected) / expected, 0.08)
                self.assertGreater(len(result["beats"]), 8)
                beat_period = float(np.median(np.diff(result["beats"])))
                grid_bpm = 60 / beat_period
                self.assertLessEqual(abs(result["bpm"] - grid_bpm) / result["bpm"], 0.1)

    def test_silence_has_no_tempo_or_beats(self) -> None:
        result = rhythm.analyze(np.zeros(rhythm.SR * 10, dtype=np.float32))
        self.assertEqual(result["bpm"], 0.0)
        self.assertEqual(len(result["beats"]), 0)


if __name__ == "__main__":
    unittest.main()
