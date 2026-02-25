from local_ocr.schemas import Element, Page, OCRResponse


def test_text_element():
    el = Element(type="text", content="hello", position=[10, 20, 100, 50])
    assert el.type == "text"
    assert el.content == "hello"
    assert el.latex is None
    assert el.image_base64 is None


def test_formula_element():
    el = Element(
        type="formula",
        content="E = mc^2",
        latex="E = mc^{2}",
        position=[10, 20, 100, 50],
    )
    assert el.latex == "E = mc^{2}"


def test_page():
    page = Page(
        page_number=1,
        elements=[Element(type="text", content="hi", position=[0, 0, 10, 10])],
    )
    assert len(page.elements) == 1


def test_ocr_response():
    resp = OCRResponse(
        pages=[Page(page_number=1, elements=[])]
    )
    assert len(resp.pages) == 1
