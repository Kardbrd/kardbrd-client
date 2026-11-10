"""Behavioral tests for the content type signed by attachment presign."""

from unittest.mock import patch

import httpx
import pytest

from kardbrd_client.client import KardbrdAPIError, KardbrdClient


@pytest.mark.parametrize(
    ("filename", "data", "requested_type", "signed_type", "use_markdown_helper"),
    [
        (
            "notes.md",
            b"# \xc3\xa9\n",
            "text/markdown",
            "text/markdown; charset=utf-8",
            False,
        ),
        (
            "data.json",
            b'{"\xc3\xa9": true}\n',
            "application/json",
            "application/json; charset=utf-8",
            False,
        ),
        (
            "notes.md",
            b"# Notes\n",
            "text/markdown",
            "text/markdown; charset=utf-8",
            True,
        ),
        (
            "latin.md",
            b"\xe9\n",
            "text/markdown; charset=iso-8859-1",
            "text/markdown; charset=iso-8859-1",
            False,
        ),
        ("image.png", b"\x89PNG\r\n\x1a\n\x00\xff", "image/png", "image/png", False),
    ],
)
def test_upload_uses_signed_content_type_and_preserves_bytes(
    tmp_path, filename, data, requested_type, signed_type, use_markdown_helper
):
    file_path = tmp_path / filename
    file_path.write_bytes(data)
    calls = []
    puts = []

    def request(method, path, json):
        calls.append((method, path, json))
        if path.endswith("/presign/"):
            return {
                "upload_url": "https://storage.test/upload",
                "s3_key": "key",
                "content_type": signed_type,
            }
        return {"id": "attachment"}

    def put(url, content, headers, timeout):
        puts.append((url, content.read(), headers["Content-Type"], timeout))
        if headers["Content-Type"] != signed_type:
            return httpx.Response(403, text="SignatureDoesNotMatch")
        return httpx.Response(200)

    with KardbrdClient("https://api.test", "synthetic-token") as client:
        with (
            patch.object(client, "_request", side_effect=request),
            patch("kardbrd_client.client.httpx.put", side_effect=put),
        ):
            if use_markdown_helper:
                result = client.upload_markdown_content(
                    "card", "notes", data.decode("utf-8")
                )
            else:
                result = client.upload_attachment(
                    "card",
                    str(file_path),
                    content_type=requested_type
                    if "charset" in requested_type
                    else None,
                )

    assert result == {"id": "attachment"}
    assert calls == [
        (
            "POST",
            "/api/cards/card/attachments/presign/",
            {
                "filename": filename,
                "content_type": requested_type,
                "file_size": len(data),
            },
        ),
        (
            "POST",
            "/api/cards/card/attachments/confirm/",
            {
                "s3_key": "key",
                "filename": filename,
                "file_size": len(data),
                "content_type": signed_type,
            },
        ),
    ]
    assert puts == [("https://storage.test/upload", data, signed_type, 300.0)]


@pytest.mark.parametrize(
    "signed_type",
    [
        "__missing__",
        None,
        "",
        "not-a-mime",
        "text/plain; charset=",
        "text/plain ",
        42,
        "text/plain\r\nInjected: true",
    ],
)
def test_invalid_presign_type_stops_before_put(tmp_path, signed_type):
    file_path = tmp_path / "notes.md"
    file_path.write_bytes(b"# Notes\n")
    calls = []

    def request(method, path, json):
        calls.append((method, path, json))
        response = {"upload_url": "https://storage.test/upload", "s3_key": "key"}
        if signed_type != "__missing__":
            response["content_type"] = signed_type
        return response

    with KardbrdClient("https://api.test", "synthetic-token") as client:
        with (
            patch.object(client, "_request", side_effect=request),
            patch("kardbrd_client.client.httpx.put") as put,
        ):
            with pytest.raises(KardbrdAPIError, match="content_type"):
                client.upload_attachment("card", str(file_path))

    assert len(calls) == 1
    assert calls[0][1].endswith("/presign/")
    put.assert_not_called()


def test_failed_put_never_confirms(tmp_path):
    file_path = tmp_path / "notes.md"
    file_path.write_bytes(b"# Notes\n")
    calls = []

    def request(method, path, json):
        calls.append((method, path, json))
        return {
            "upload_url": "https://storage.test/upload",
            "s3_key": "key",
            "content_type": "text/markdown; charset=utf-8",
        }

    with KardbrdClient("https://api.test", "synthetic-token") as client:
        with (
            patch.object(client, "_request", side_effect=request),
            patch(
                "kardbrd_client.client.httpx.put",
                return_value=httpx.Response(403, text="denied"),
            ) as put,
        ):
            with pytest.raises(KardbrdAPIError) as error:
                client.upload_attachment("card", str(file_path))

    assert error.value.status_code == 403
    assert len(calls) == 1
    assert calls[0][1].endswith("/presign/")
    assert put.call_count == 1


def test_retry_reuses_signed_header_url_and_bytes_then_confirms_once(tmp_path):
    data = b"# Notes\n\x00\xff"
    file_path = tmp_path / "notes.md"
    file_path.write_bytes(data)
    calls = []
    puts = []
    signed_type = "text/markdown; charset=utf-8"

    def request(method, path, json):
        calls.append((method, path, json))
        if path.endswith("/presign/"):
            return {
                "upload_url": "https://storage.test/upload",
                "s3_key": "key",
                "content_type": signed_type,
            }
        return {"id": "attachment"}

    def put(url, content, headers, timeout):
        puts.append((url, content.read(), headers["Content-Type"]))
        return httpx.Response(
            500 if len(puts) == 1 else 200, text="retry" if len(puts) == 1 else ""
        )

    with KardbrdClient("https://api.test", "synthetic-token") as client:
        with (
            patch.object(client, "_request", side_effect=request),
            patch("kardbrd_client.client.httpx.put", side_effect=put),
        ):
            result = client.upload_attachment("card", str(file_path))

    assert result == {"id": "attachment"}
    assert puts == [("https://storage.test/upload", data, signed_type)] * 2
    assert calls == [
        (
            "POST",
            "/api/cards/card/attachments/presign/",
            {
                "filename": "notes.md",
                "content_type": "text/markdown",
                "file_size": len(data),
            },
        ),
        (
            "POST",
            "/api/cards/card/attachments/confirm/",
            {
                "s3_key": "key",
                "filename": "notes.md",
                "file_size": len(data),
                "content_type": signed_type,
            },
        ),
    ]
