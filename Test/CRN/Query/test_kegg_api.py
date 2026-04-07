from __future__ import annotations

import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import requests

from synkit.CRN.Query.kegg_api import KEGGClient


class _KEGGTestHandler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:  # noqa: N802
        if self.path == "/ok":
            body = b"ENTRY       hsa00010\nNAME        Glycolysis / Gluconeogenesis\n"
            self.send_response(200)
            self.send_header("Content-Type", "text/plain; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return

        if self.path == "/empty":
            body = b""
            self.send_response(200)
            self.send_header("Content-Type", "text/plain; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return

        self.send_response(404)
        self.send_header("Content-Type", "text/plain; charset=utf-8")
        self.end_headers()
        self.wfile.write(b"Not Found")

    def log_message(self, format: str, *args) -> None:  # noqa: A003
        return


class TestKEGGClient(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.server = ThreadingHTTPServer(("127.0.0.1", 0), _KEGGTestHandler)
        host, port = cls.server.server_address
        cls.base_url = f"http://{host}:{port}"
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()

    @classmethod
    def tearDownClass(cls) -> None:
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join(timeout=2)

    def test_get_text_returns_response_text(self) -> None:
        client = KEGGClient(base_url=self.base_url, timeout=5.0)
        text = client.get_text("ok")

        self.assertIn("ENTRY", text)
        self.assertIn("hsa00010", text)

    def test_get_text_supports_empty_body(self) -> None:
        client = KEGGClient(base_url=self.base_url, timeout=5.0)
        text = client.get_text("empty")

        self.assertEqual(text, "")

    def test_get_text_raises_http_error_on_failure(self) -> None:
        client = KEGGClient(base_url=self.base_url, timeout=5.0)

        with self.assertRaises(requests.HTTPError):
            client.get_text("missing")

    def test_get_optional_text_returns_text_when_available(self) -> None:
        client = KEGGClient(base_url=self.base_url, timeout=5.0)
        text = client.get_optional_text("ok")

        self.assertIsInstance(text, str)
        self.assertIn("Glycolysis", text)

    def test_get_optional_text_returns_none_on_http_error(self) -> None:
        client = KEGGClient(base_url=self.base_url, timeout=5.0)
        text = client.get_optional_text("missing")

        self.assertIsNone(text)


if __name__ == "__main__":
    unittest.main()
