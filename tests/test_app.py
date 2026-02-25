from unittest.mock import patch, MagicMock
import pytest
from httpx import AsyncClient, ASGITransport
from PIL import Image
import io

from local_ocr.schemas import Element


def _make_test_image_bytes() -> bytes:
    img = Image.new("RGB", (100, 100), "white")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _mock_elements():
    return [
        Element(type="text", content="hello", position=[0, 0, 50, 20], score=0.95),
        Element(
            type="formula",
            content="x^2",
            latex="x^{2}",
            position=[0, 30, 50, 20],
            score=0.9,
        ),
    ]


@pytest.fixture
def mock_engine():
    with patch("local_ocr.app.engine") as mock:
        mock.recognize_image.return_value = _mock_elements()
        yield mock


@pytest.mark.asyncio
async def test_health():
    from local_ocr.app import app

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


@pytest.mark.asyncio
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
    assert len(data["pages"][0]["elements"]) == 2
    assert data["pages"][0]["elements"][0]["type"] == "text"
    assert data["pages"][0]["elements"][1]["latex"] == "x^{2}"


@pytest.mark.asyncio
async def test_ocr_pdf(mock_engine):
    from local_ocr.app import app

    with patch("local_ocr.app.pdf_to_images") as mock_pdf:
        mock_pdf.return_value = [Image.new("RGB", (100, 100), "white")]
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post(
                "/ocr/pdf",
                files={"file": ("test.pdf", b"%PDF-fake", "application/pdf")},
            )
    assert resp.status_code == 200
    data = resp.json()
    assert len(data["pages"]) == 1


@pytest.mark.asyncio
async def test_ocr_url_image(mock_engine):
    from local_ocr.app import app

    with patch("local_ocr.app.fetch_url") as mock_fetch:
        mock_fetch.return_value = {
            "type": "image",
            "data": Image.new("RGB", (100, 100), "white"),
        }
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post("/ocr/url", json={"url": "http://example.com/img.png"})
    assert resp.status_code == 200
    data = resp.json()
    assert len(data["pages"]) == 1
