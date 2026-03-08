# tests/test_app.py
from unittest.mock import patch, MagicMock
import pytest
from httpx import AsyncClient, ASGITransport
from PIL import Image
import io


def _make_test_image_bytes() -> bytes:
    img = Image.new("RGB", (100, 100), "white")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


MOCK_MARKDOWN = "# Question 1\n\nSolve $x^2 = 4$\n\n$$x = \\pm 2$$"


@pytest.fixture
def mock_engine():
    with patch("local_ocr.app.engine") as mock:
        mock.recognize_image.return_value = MOCK_MARKDOWN
        yield mock


async def test_health():
    from local_ocr.app import app

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"


async def test_ocr_image(mock_engine):
    from local_ocr.app import app

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post(
            "/ocr/image",
            files={"file": ("test.png", _make_test_image_bytes(), "image/png")},
        )
    assert resp.status_code == 200
    data = resp.json()
    assert len(data["pages"]) == 1
    assert data["pages"][0]["page_number"] == 1
    assert "$x^2 = 4$" in data["pages"][0]["markdown"]


async def test_ocr_pdf(mock_engine):
    from local_ocr.app import app

    with patch("local_ocr.app.pdf_to_images") as mock_pdf:
        mock_pdf.return_value = [
            Image.new("RGB", (100, 100), "white"),
            Image.new("RGB", (100, 100), "white"),
        ]
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post(
                "/ocr/pdf",
                files={"file": ("test.pdf", b"%PDF-fake", "application/pdf")},
            )
    assert resp.status_code == 200
    data = resp.json()
    assert len(data["pages"]) == 2
    assert data["pages"][0]["page_number"] == 1
    assert data["pages"][1]["page_number"] == 2
    assert "$x^2 = 4$" in data["pages"][0]["markdown"]


async def test_ocr_url_image(mock_engine):
    from local_ocr.app import app

    with patch("local_ocr.app.fetch_url") as mock_fetch:
        mock_fetch.return_value = {
            "type": "image",
            "data": Image.new("RGB", (100, 100), "white"),
        }
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post(
                "/ocr/url", json={"url": "http://example.com/img.png"}
            )
    assert resp.status_code == 200
    data = resp.json()
    assert len(data["pages"]) == 1
    assert "$x^2 = 4$" in data["pages"][0]["markdown"]


async def test_ocr_url_pdf(mock_engine):
    from local_ocr.app import app

    with patch("local_ocr.app.fetch_url") as mock_fetch:
        mock_fetch.return_value = {"type": "pdf", "data": b"%PDF-fake"}

        with patch("local_ocr.app.pdf_to_images") as mock_pdf:
            mock_pdf.return_value = [Image.new("RGB", (100, 100), "white")]
            transport = ASGITransport(app=app)
            async with AsyncClient(
                transport=transport, base_url="http://test"
            ) as client:
                resp = await client.post(
                    "/ocr/url", json={"url": "http://example.com/doc.pdf"}
                )
    assert resp.status_code == 200
    data = resp.json()
    assert len(data["pages"]) == 1


async def test_ocr_url_bad_url():
    from local_ocr.app import app

    with patch("local_ocr.app.engine"):
        with patch("local_ocr.app.fetch_url", side_effect=Exception("Connection error")):
            transport = ASGITransport(app=app)
            async with AsyncClient(
                transport=transport, base_url="http://test"
            ) as client:
                resp = await client.post(
                    "/ocr/url", json={"url": "http://bad-url.example.com"}
                )
    assert resp.status_code == 400
