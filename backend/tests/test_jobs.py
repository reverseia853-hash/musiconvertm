import asyncio
import tempfile
import unittest
import uuid
from pathlib import Path
from threading import Event
from unittest.mock import patch

import main
import pipeline


class JobCancellationTests(unittest.TestCase):
    def test_application_startup_preserves_uploaded_audio(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            upload_dir = Path(temp_dir) / "uploads"
            upload_dir.mkdir()
            audio = upload_dir / "uploaded-audio.mp3"
            audio.write_bytes(b"keep this upload")

            async def start_application() -> None:
                async with main.lifespan(main.app):
                    self.assertEqual(audio.read_bytes(), b"keep this upload")

            with patch.object(main, "UPLOADS_DIR", upload_dir), patch.object(
                main, "MAX_UPLOAD_BYTES", 1
            ), patch.object(main, "AUTH_REQUIRED", False):
                asyncio.run(start_application())

    def test_cancellation_endpoint_marks_an_active_job(self) -> None:
        job_id = uuid.uuid4().hex
        main.jobs[job_id] = {"status": "transcribing"}
        main.job_cancel_events[job_id] = Event()
        try:
            result = main.cancel_job(job_id)

            self.assertTrue(result["accepted"])
            self.assertEqual(result["status"], "cancelling")
            self.assertTrue(main.job_cancel_events[job_id].is_set())
        finally:
            main.jobs.pop(job_id, None)
            main.job_cancel_events.pop(job_id, None)

    def test_cancelled_job_waiting_for_worker_lock_does_not_start(self) -> None:
        event = Event()
        event.set()

        with self.assertRaises(pipeline.JobCancelled):
            with main.cancellable_run_lock(event):
                self.fail("A cancelled job must not acquire the worker lock.")

    def test_worker_finishes_with_cancelled_status(self) -> None:
        job_id = uuid.uuid4().hex
        with tempfile.TemporaryDirectory() as temp_dir:
            main.jobs[job_id] = {"status": "queued", "progress": 0.0}
            main.job_cancel_events[job_id] = Event()
            try:
                with patch("main.get_model", return_value=object()), patch(
                    "main.pipeline.transcribe_mode", side_effect=pipeline.JobCancelled
                ):
                    main.run_job(
                        job_id, Path(temp_dir), Path(temp_dir) / "audio.mp3",
                        "", False, "vad", "small", False,
                    )

                self.assertEqual(main.jobs[job_id]["status"], "cancelled")
            finally:
                main.jobs.pop(job_id, None)
                main.job_cancel_events.pop(job_id, None)


if __name__ == "__main__":
    unittest.main()
