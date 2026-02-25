from unittest.mock import patch, MagicMock
from PIL import Image
from local_ocr.input_handler import pdf_to_images, fetch_url


def test_pdf_to_images_returns_pil_images(tmp_path):
    """Use a tiny valid PDF to test conversion."""
    import fitz

    doc = fitz.open()
    page = doc.new_page(width=100, height=100)
    page.insert_text((10, 50), "test")
    pdf_path = tmp_path / "test.pdf"
    doc.save(str(pdf_path))
    doc.close()

    images = pdf_to_images(pdf_path)
    assert len(images) == 1
    assert isinstance(images[0], Image.Image)


def test_pdf_to_images_specific_pages(tmp_path):
    import fitz

    doc = fitz.open()
    for i in range(3):
        page = doc.new_page(width=100, height=100)
        page.insert_text((10, 50), f"page {i}")
    pdf_path = tmp_path / "test.pdf"
    doc.save(str(pdf_path))
    doc.close()

    images = pdf_to_images(pdf_path, page_numbers=[0, 2])
    assert len(images) == 2


@patch("local_ocr.input_handler.httpx")
def test_fetch_url_image(mock_httpx):
    img = Image.new("RGB", (10, 10), "red")
    import io

    buf = io.BytesIO()
    img.save(buf, format="PNG")
    png_bytes = buf.getvalue()

    mock_response = MagicMock()
    mock_response.content = png_bytes
    mock_response.headers = {"content-type": "image/png"}
    mock_response.raise_for_status = MagicMock()
    mock_httpx.get.return_value = mock_response

    result = fetch_url("http://example.com/img.png")
    assert result["type"] == "image"
    assert isinstance(result["data"], Image.Image)


@patch("local_ocr.input_handler.httpx")
def test_fetch_url_pdf(mock_httpx, tmp_path):
    mock_response = MagicMock()
    mock_response.content = b"%PDF-fake"
    mock_response.headers = {"content-type": "application/pdf"}
    mock_response.raise_for_status = MagicMock()
    mock_httpx.get.return_value = mock_response

    result = fetch_url("http://example.com/doc.pdf")
    assert result["type"] == "pdf"
    assert isinstance(result["data"], bytes)
