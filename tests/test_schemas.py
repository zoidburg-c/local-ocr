# tests/test_schemas.py
from local_ocr.schemas import Page, OCRResponse, URLRequest


def test_page_with_markdown():
    page = Page(page_number=1, markdown="# Title\n\nSome text with $x^2$")
    assert page.page_number == 1
    assert "$x^2$" in page.markdown


def test_ocr_response():
    resp = OCRResponse(
        pages=[Page(page_number=1, markdown="Hello")]
    )
    assert len(resp.pages) == 1
    assert resp.pages[0].markdown == "Hello"


def test_url_request():
    req = URLRequest(url="http://example.com/img.png")
    assert req.url == "http://example.com/img.png"
