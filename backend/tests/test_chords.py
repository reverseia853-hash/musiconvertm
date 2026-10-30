import unittest
from unittest.mock import patch

import numpy as np

import chords


class ChordDetectionTests(unittest.TestCase):
    @staticmethod
    def chromatic_triads() -> tuple[np.ndarray, list[str]]:
        sample_rate = chords.SR
        duration = 3
        time = np.arange(sample_rate * duration, dtype=np.float32) / sample_rate
        fade = np.minimum(1, time / 0.04) * np.minimum(1, (duration - time) / 0.04)
        blocks = []
        expected = []

        for root in range(12):
            for minor, third in ((False, 4), (True, 3)):
                notes = (root, (root + third) % 12, (root + 7) % 12)
                signal = np.zeros_like(time)
                for note in notes:
                    frequency = 440 * 2 ** ((note - 9) / 12)
                    signal += np.sin(2 * np.pi * frequency * time)
                    signal += 0.25 * np.sin(4 * np.pi * frequency * time)
                blocks.append((0.2 * signal * fade).astype(np.float32))
                expected.append(chords.NOTES[root] + ("m" if minor else ""))

        return np.concatenate(blocks), expected

    @staticmethod
    def chromatic_sevenths() -> tuple[np.ndarray, list[str]]:
        sample_rate = chords.SR
        duration = 2.5
        time = np.arange(round(sample_rate * duration), dtype=np.float32) / sample_rate
        fade = np.minimum(1, time / 0.04) * np.minimum(1, (duration - time) / 0.04)
        qualities = (
            ("7", (0, 4, 7, 10)),
            ("maj7", (0, 4, 7, 11)),
            ("m7", (0, 3, 7, 10)),
        )
        blocks = []
        expected = []

        for root in range(12):
            for suffix, intervals in qualities:
                signal = np.zeros_like(time)
                for interval in intervals:
                    note = (root + interval) % 12
                    frequency = 440 * 2 ** ((note - 9) / 12)
                    signal += np.sin(2 * np.pi * frequency * time)
                    signal += 0.25 * np.sin(4 * np.pi * frequency * time)
                blocks.append((0.2 * signal * fade).astype(np.float32))
                expected.append(chords.NOTES[root] + suffix)

        return np.concatenate(blocks), expected

    def test_detects_at_least_ninety_percent_of_synthetic_major_and_minor_triads(self) -> None:
        audio, expected = self.chromatic_triads()
        result = chords.detect_signal(audio, sr=chords.SR)
        correct = 0

        for index, chord in enumerate(expected):
            center = (index + 0.5) * 3
            segment = next(
                (item for item in result["chords"] if item["start"] <= center < item["end"]),
                None,
            )
            if segment and segment["chord"] == chord:
                correct += 1
            if segment:
                self.assertGreaterEqual(segment["conf"], 0.0)
                self.assertLessEqual(segment["conf"], 1.0)

        self.assertGreaterEqual(correct / len(expected), 0.9)
        self.assertFalse(any(item["chord"].endswith("7") for item in result["chords"]))

    def test_detects_seventh_qualities_in_all_keys(self) -> None:
        audio, expected = self.chromatic_sevenths()
        result = chords.detect_signal(audio, sr=chords.SR, use_beats=False)
        detected = []

        for index, chord in enumerate(expected):
            center = (index + 0.5) * 2.5
            segment = next(
                (item for item in result["chords"] if item["start"] <= center < item["end"]),
                None,
            )
            detected.append(segment["chord"] if segment else None)

        correct = sum(actual == wanted for actual, wanted in zip(detected, expected))
        self.assertGreaterEqual(correct / len(expected), 0.9, list(zip(expected, detected)))

    def test_faint_seventh_does_not_override_a_major_triad(self) -> None:
        duration = 5
        time = np.arange(chords.SR * duration, dtype=np.float32) / chords.SR
        fade = np.minimum(1, time / 0.04) * np.minimum(1, (duration - time) / 0.04)
        audio = np.zeros_like(time)
        for interval, amplitude in ((0, 1.0), (4, 1.0), (7, 1.0), (10, 0.12)):
            frequency = 440 * 2 ** (((interval - 9) % 12) / 12)
            audio += amplitude * np.sin(2 * np.pi * frequency * time)
            audio += 0.25 * amplitude * np.sin(4 * np.pi * frequency * time)
        audio = (0.2 * audio * fade).astype(np.float32)

        result = chords.detect_signal(audio, sr=chords.SR, use_beats=False)

        segment = next(item for item in result["chords"] if item["start"] <= 2.5 < item["end"])
        self.assertEqual(segment["chord"], "C")

    def test_repeated_context_corrects_one_noisy_chord_interval(self) -> None:
        progression = np.eye(4, dtype=np.float32)[[0, 1, 2, 3] * 6]
        expected = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        progression[8] = np.array([0.5, 0.5, 0.0, 0.0], dtype=np.float32)

        before = float(
            progression[8] @ expected
            / (np.linalg.norm(progression[8]) * np.linalg.norm(expected))
        )
        smoothed = chords._nonlocal_smooth(progression)
        after = float(
            smoothed[8] @ expected
            / (np.linalg.norm(smoothed[8]) * np.linalg.norm(expected))
        )

        self.assertGreater(after, before)
        self.assertGreater(after, 0.95)

    def test_preserves_chords_that_change_on_consecutive_beats(self) -> None:
        bpm = 145.0
        beat = 60 / bpm
        progression = ["C", "G", "Am", "Em", "F", "G", "E", "Am"] * 2
        notes = {
            "C": (0, 4, 7),
            "G": (7, 11, 2),
            "Am": (9, 0, 4),
            "Em": (4, 7, 11),
            "F": (5, 9, 0),
            "E": (4, 8, 11),
        }
        sample_count = round(len(progression) * beat * chords.SR)
        time = np.arange(sample_count, dtype=np.float32) / chords.SR
        audio = np.zeros(sample_count, dtype=np.float32)

        for index, chord in enumerate(progression):
            start = round(index * beat * chords.SR)
            end = min(round((index + 1) * beat * chords.SR), sample_count)
            local_time = time[start:end] - index * beat
            chord_duration = (end - start) / chords.SR
            fade = np.minimum(1, np.minimum(local_time / 0.03, (chord_duration - local_time) / 0.03))
            for note in notes[chord]:
                frequency = 440 * 2 ** ((note - 9) / 12)
                tone = np.sin(2 * np.pi * frequency * time[start:end])
                tone += 0.25 * np.sin(4 * np.pi * frequency * time[start:end])
                audio[start:end] += 0.2 * tone * fade

        beats = np.arange(len(progression), dtype=np.float64) * beat
        with patch.object(
            chords.rhythm,
            "analyze",
            return_value={"bpm": bpm, "beats": beats},
        ):
            result = chords.detect_signal(audio, sr=chords.SR)

        detected = []
        for index in range(1, len(progression)):
            center = (index + 0.5) * beat
            segment = next(
                (item for item in result["chords"] if item["start"] <= center < item["end"]),
                None,
            )
            detected.append(segment["chord"] if segment else None)

        self.assertGreaterEqual(len(result["chords"]), len(progression) - 1)
        self.assertGreaterEqual(
            sum(actual == expected for actual, expected in zip(detected, progression[1:])),
            len(progression) - 2,
        )


if __name__ == "__main__":
    unittest.main()
