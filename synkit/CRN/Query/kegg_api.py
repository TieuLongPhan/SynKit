from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import requests


@dataclass(slots=True)
class KEGGClient:
    """
    Lightweight REST client for the KEGG API.

    This client provides a minimal wrapper around the KEGG REST interface and
    returns raw response text for a requested endpoint.

    :param base_url:
        Base KEGG REST endpoint.
    :type base_url: str
    :param timeout:
        Request timeout in seconds.
    :type timeout: float

    Example
    -------
    .. code-block:: python

        client = KEGGClient(
            base_url="https://rest.kegg.jp",
            timeout=30.0,
        )
        text = client.get_text("get/hsa00010")
    """

    base_url: str = "https://rest.kegg.jp"
    timeout: float = 60.0

    def get_text(self, path: str) -> str:
        """
        Send ``GET <base_url>/<path>`` and return the response body as text.

        :param path:
            Relative KEGG REST path, for example ``"get/hsa00010"``.
        :type path: str

        :returns:
            Raw text returned by the KEGG REST API.
        :rtype: str

        :raises requests.HTTPError:
            Raised when the server responds with an HTTP error status.
        :raises requests.RequestException:
            Raised for transport-level request failures.

        Example
        -------
        .. code-block:: python

            client = KEGGClient()
            entry_text = client.get_text("get/hsa00010")
        """
        response: requests.Response = requests.get(
            f"{self.base_url}/{path}",
            timeout=self.timeout,
        )
        response.raise_for_status()
        return response.text

    def get_optional_text(self, path: str) -> Optional[str]:
        """
        Send a GET request and return the response text when available.

        Unlike :meth:`get_text`, this method suppresses
        :class:`requests.HTTPError` and returns ``None`` for HTTP-level
        failures such as ``404 Not Found``.

        :param path:
            Relative KEGG REST path.
        :type path: str

        :returns:
            Response text if the request succeeds, otherwise ``None`` when an
            HTTP error occurs.
        :rtype: Optional[str]

        :raises requests.RequestException:
            Raised for non-HTTP request failures, such as connection errors or
            timeouts.

        Example
        -------
        .. code-block:: python

            client = KEGGClient()
            maybe_text = client.get_optional_text("get/hsa00010")
            if maybe_text is not None:
                print(maybe_text[:200])
        """
        try:
            return self.get_text(path)
        except requests.HTTPError:
            return None
