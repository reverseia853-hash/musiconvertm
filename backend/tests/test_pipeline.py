import unittest
import tempfile
from pathlib import Path
from threading import Event
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pipeline


class TranscriptionPipelineTests(unittest.TestCase):
    @staticmethod
    def fake_model() -> Mock:
        word = SimpleNamespace(word=" hola", start=0.5, end=1.0)
        segment = SimpleNamespace(start=0.5, end=1.0, text="hola", words=[word])
        model = Mock()
        model.transcribe.return_value = (iter([segment]), SimpleNamespace(duration=4.0))
        return model

    def test_singing_transcription_uses_sensitive_vad_and_silence_guard(self) -> None:
        model = self.fake_model()
        progress = []

        result = pipeline.transcribe(model, Path("song.mp3"), "es", 3, progress.append)

        options = model.transcribe.call_args.kwargs
        self.assertTrue(options["vad_filter"])
        self.assertEqual(options["vad_parameters"], {"threshold": 0.15, "min_silence_duration_ms": 500})
        self.assertEqual(options["no_speech_threshold"], 0.6)
        self.assertEqual(result[0]["words"][0]["word"], "hola")
        self.assertEqual(progress, [0.25])

    def test_vad_disabled_pass_still_rejects_high_no_speech_segments(self) -> None:
        model = self.fake_model()

        pipeline.transcribe(model, Path("song.mp3"), "es", 5, lambda _: None, vad=False)

        options = model.transcribe.call_args.kwargs
        self.assertFalse(options["vad_filter"])
        self.assertIsNone(options["vad_parameters"])
        self.assertEqual(options["no_speech_threshold"], 0.6)

    def test_no_vad_mode_uses_higher_beam_size(self) -> None:
        model = self.fake_model()

        result, alternative = pipeline.transcribe_mode(
            model, Path("song.mp3"), "es", 3, lambda _: None, "no_vad"
        )

        self.assertIsNone(alternative)
        self.assertEqual(result[0]["words"][0]["word"], "hola")
        self.assertEqual(model.transcribe.call_args.kwargs["beam_size"], 5)
        self.assertFalse(model.transcribe.call_args.kwargs["vad_filter"])

    def test_compare_mode_returns_both_transcripts_and_reports_progress(self) -> None:
        model = Mock()
        model.transcribe.side_effect = lambda *_args, **_kwargs: self.fake_model().transcribe()
        progress = []
        stages = []

        result, alternative = pipeline.transcribe_mode(
            model, Path("song.mp3"), "es", 3, progress.append, "compare", stages.append
        )

        self.assertEqual(result[0]["words"][0]["word"], "hola")
        self.assertEqual(alternative[0]["words"][0]["word"], "hola")
        self.assertEqual(model.transcribe.call_count, 2)
        self.assertTrue(model.transcribe.call_args_list[0].kwargs["vad_filter"])
        self.assertFalse(model.transcribe.call_args_list[1].kwargs["vad_filter"])
        self.assertEqual(model.transcribe.call_args_list[1].kwargs["beam_size"], 5)
        self.assertEqual(stages, ["transcribing", "transcribing_alternative"])
        self.assertEqual(progress, [0.125, 0.625])

    def test_unknown_transcription_mode_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            pipeline.transcribe_mode(
                self.fake_model(), Path("song.mp3"), "es", 3, lambda _: None, "unknown"
            )

    def test_transcription_stops_when_cancellation_is_requested(self) -> None:
        with self.assertRaises(pipeline.JobCancelled):
            pipeline.transcribe(
                self.fake_model(), Path("song.mp3"), "es", 3, lambda _: None,
                is_cancelled=lambda: True,
            )

    def test_separation_levels_configure_demucs_quality_and_runtime(self) -> None:
        expected = {
            "low": ("1", "0.1"),
            "medium": ("2", "0.25"),
            "high": ("4", "0.5"),
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            job_dir = Path(temp_dir)
            vocals = job_dir / "separated" / "htdemucs" / "song" / "vocals.mp3"
            vocals.parent.mkdir(parents=True)
            vocals.touch()
            for level, (shifts, overlap) in expected.items():
                process = Mock()
                process.stdout.read1.return_value = b""
                process.wait.return_value = 0
                process.returncode = 0
                with self.subTest(level=level), patch(
                    "pipeline.subprocess.Popen", return_value=process
                ) as popen:
                    result = pipeline.separate_vocals(
                        Path("song.mp3"), job_dir, lambda _: None,
                        separation_level=level,
                    )

                    command = popen.call_args.args[0]
                    self.assertEqual(command[command.index("--shifts") + 1], shifts)
                    self.assertEqual(command[command.index("--overlap") + 1], overlap)
                    self.assertEqual(result, vocals)

    def test_invalid_separation_level_is_rejected_before_launch(self) -> None:
        with patch("pipeline.subprocess.Popen") as popen:
            with self.assertRaises(ValueError):
                pipeline.separate_vocals(
                    Path("song.mp3"), Path("job"), lambda _: None,
                    separation_level="extreme",
                )

        popen.assert_not_called()

    def test_cancelling_demucs_terminates_its_subprocess(self) -> None:
        cancel_event = Event()
        process = Mock()
        process.stdout.read1.return_value = b"10%|"
        process.poll.return_value = None
        with tempfile.TemporaryDirectory() as temp_dir, patch(
            "pipeline.subprocess.Popen", return_value=process
        ):
            with self.assertRaises(pipeline.JobCancelled):
                pipeline.separate_vocals(
                    Path("song.mp3"), Path(temp_dir), lambda _: cancel_event.set(),
                    is_cancelled=cancel_event.is_set,
                )

        process.terminate.assert_called_once()
        process.wait.assert_called_once_with(timeout=5)


if __name__ == "__main__":
    unittest.main()
