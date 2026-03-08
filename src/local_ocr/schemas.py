# src/local_ocr/schemas.py
from pydantic import BaseModel


class Page(BaseModel):
    page_number: int
    markdown: str


class OCRResponse(BaseModel):
    pages: list[Page]


class URLRequest(BaseModel):
    url: str
