import io
from pathlib import Path

import fitz
import httpx
from PIL import Image


def pdf_to_images(
    pdf_path: str | Path, page_numbers: list[int] | None = None, dpi: int = 300
) -> list[Image.Image]:
    """Convert PDF pages to PIL Images."""
    doc = fitz.open(str(pdf_path))
    pages = page_numbers if page_numbers is not None else range(len(doc))
    images = []
    zoom = dpi / 72
    matrix = fitz.Matrix(zoom, zoom)
    for page_num in pages:
        page = doc[page_num]
        pix = page.get_pixmap(matrix=matrix)
        img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
        images.append(img)
    doc.close()
    return images


def fetch_url(url: str) -> dict:
    """Fetch a URL and return its content with type info.

    Returns:
        {"type": "image", "data": PIL.Image} or
        {"type": "pdf", "data": bytes}
    """
    response = httpx.get(url, follow_redirects=True, timeout=30.0)
    response.raise_for_status()
    content_type = response.headers.get("content-type", "")

    if "pdf" in content_type:
        return {"type": "pdf", "data": response.content}

    # Assume image for everything else
    img = Image.open(io.BytesIO(response.content))
    return {"type": "image", "data": img}
