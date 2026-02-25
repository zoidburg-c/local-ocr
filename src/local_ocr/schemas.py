from pydantic import BaseModel


class Element(BaseModel):
    type: str  # "text", "formula", "diagram"
    content: str | None = None
    latex: str | None = None
    image_base64: str | None = None
    position: list[float]  # [x, y, w, h]
    score: float | None = None


class Page(BaseModel):
    page_number: int
    elements: list[Element]


class OCRResponse(BaseModel):
    pages: list[Page]


class URLRequest(BaseModel):
    url: str
