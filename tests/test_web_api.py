from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from fastapi.testclient import TestClient

from facade_synth.web import create_app


class WebStudioApiTests(unittest.TestCase):
    def test_a_producer_can_create_and_confirm_a_generation_brief(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            client = TestClient(create_app(workspace=Path(temp_dir)))
            response = client.post(
                "/api/jobs",
                json={
                    "task": "building_use",
                    "output_target": 3,
                    "split_ratio": {"train": 1.0, "validation": 0.0, "test": 0.0},
                    "building_use_distribution": {"office": 1.0},
                    "render_width": 64,
                    "render_height": 64,
                    "seed": 11,
                },
            )
            self.assertEqual(201, response.status_code)
            job_id = response.json()["id"]
            self.assertEqual("draft", response.json()["state"])

            confirmation = client.post(
                f"/api/jobs/{job_id}/confirm",
                json={"confirmed_by": "leo"},
            )
            self.assertEqual(200, confirmation.status_code)
            self.assertEqual("queued", confirmation.json()["state"])

            page = client.get("/")
            self.assertEqual(200, page.status_code)
            self.assertIn("Generation Brief", page.text)
            self.assertIn("Lighting intensity minimum", page.text)

    def test_review_rejects_non_boolean_approval(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            client = TestClient(create_app(workspace=Path(temp_dir)))
            created = client.post(
                "/api/jobs",
                json={
                    "task": "building_use",
                    "output_target": 3,
                    "split_ratio": {"train": 1.0, "validation": 0.0, "test": 0.0},
                    "building_use_distribution": {"office": 1.0},
                    "render_width": 64,
                    "render_height": 64,
                },
            ).json()
            job_id = created["id"]
            response = client.post(
                f"/api/jobs/{job_id}/review",
                json={"reviewer": "leo", "approved": "false"},
            )

            self.assertEqual(422, response.status_code)


if __name__ == "__main__":
    unittest.main()
