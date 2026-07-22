from __future__ import annotations

import unittest

from dataclasses import dataclass
import requests

from synkit.CRN.Query.kegg_api import KEGGClient


@dataclass(frozen=True)
class _FakeResponse:
    status_code: int
    text: str = ""

    def raise_for_status(self) -> None:
        if 400 <= self.status_code < 600:
            raise requests.HTTPError(f"{self.status_code} Client Error")


def _fake_requests_get(url: str, timeout: float | None = None) -> _FakeResponse:
    if url.endswith("/ok"):
        return _FakeResponse(
            200,
            "ENTRY       hsa00010\nNAME        Glycolysis / Gluconeogenesis\n",
        )
    if url.endswith("/empty"):
        return _FakeResponse(200, "")
    return _FakeResponse(404, "Not Found")


class TestKEGGClient(unittest.TestCase):
    _base_url = "http://localhost:0"

    def test_get_text_returns_response_text(self) -> None:
        client = KEGGClient(base_url=self._base_url, timeout=5.0)
        text = client.get_text("ok")

        self.assertIn("ENTRY", text)
        self.assertIn("hsa00010", text)

    def test_get_text_supports_empty_body(self) -> None:
        client = KEGGClient(base_url=self._base_url, timeout=5.0)
        text = client.get_text("empty")

        self.assertEqual(text, "")

    def test_get_text_raises_http_error_on_failure(self) -> None:
        client = KEGGClient(base_url=self._base_url, timeout=5.0)

        with self.assertRaises(requests.HTTPError):
            client.get_text("missing")

    def test_get_optional_text_returns_text_when_available(self) -> None:
        client = KEGGClient(base_url=self._base_url, timeout=5.0)
        text = client.get_optional_text("ok")

        self.assertIsInstance(text, str)
        self.assertIn("Glycolysis", text)

    def test_get_optional_text_returns_none_on_http_error(self) -> None:
        client = KEGGClient(base_url=self._base_url, timeout=5.0)
        text = client.get_optional_text("missing")

        self.assertIsNone(text)

    @classmethod
    def setUpClass(cls) -> None:
        cls._requests_get = requests.get
        requests.get = _fake_requests_get  # type: ignore[assignment]

    @classmethod
    def tearDownClass(cls) -> None:
        requests.get = cls._requests_get  # type: ignore[assignment]


if __name__ == "__main__":
    unittest.main()
