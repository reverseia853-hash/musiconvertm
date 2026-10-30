import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient

import main


class ProgressiveWebAppTests(unittest.TestCase):
    def test_install_assets_are_public_but_api_stays_protected(self) -> None:
        with patch.object(main, "AUTH_REQUIRED", True), patch.object(
            main, "APP_PASSWORD", "test-password-long-enough"
        ), patch.object(main, "APP_SESSION_SECRET", "test-session-secret-long-enough-for-tests"):
            with TestClient(main.app) as client:
                for path in (
                    "/manifest.webmanifest",
                    "/sw.js",
                    "/icons/icon-192.png",
                    "/icons/icon-512.png",
                ):
                    with self.subTest(path=path):
                        response = client.get(path)
                        self.assertEqual(response.status_code, 200)

                protected_response = client.get("/api/jobs/not-a-real-job")
                self.assertEqual(protected_response.status_code, 401)


if __name__ == "__main__":
    unittest.main()
