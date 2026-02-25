# Local OCR Service — Design

## Purpose

A local REST API service for extracting math equations (as LaTeX), text, and diagrams from images and PDFs. Primary use case is content digitalization of mathematical and diagrammatic material.

## Architecture

FastAPI REST service wrapping Pix2Text for OCR. Three input modes: image upload, PDF upload, and URL fetch. All return structured JSON with text blocks, math equations (LaTeX), and diagram regions.

```
Client → FastAPI → Input Handler (image/PDF/URL) → Pix2Text engine → Structured JSON response
```

The Pix2Text model loads once at startup and is reused across requests. PDF pages are converted to images via PyMuPDF before processing. URLs are fetched with httpx, then routed through the same image/PDF pipeline.

## API Endpoints

- `POST /ocr/image` — Upload an image file (PNG, JPG, WEBP, TIFF). Returns OCR results for that single image.
- `POST /ocr/pdf` — Upload a PDF file. Returns OCR results per page.
- `POST /ocr/url` — JSON body with a URL. Fetches the resource (image or PDF), then processes it.
- `GET /health` — Health check, confirms model is loaded.

## Response Schema

```json
{
  "pages": [
    {
      "page_number": 1,
      "elements": [
        {"type": "text", "content": "The equation is:", "position": [x, y, w, h]},
        {"type": "formula", "content": "E = mc^2", "latex": "E = mc^{2}", "position": [x, y, w, h]},
        {"type": "diagram", "image_base64": "...", "position": [x, y, w, h]}
      ]
    }
  ]
}
```

Single-image requests return one page. PDFs return multiple.

## Project Structure

```
local-ocr/
├── pyproject.toml
├── src/
│   └── local_ocr/
│       ├── __init__.py
│       ├── app.py          # FastAPI app, endpoints
│       ├── engine.py        # Pix2Text wrapper, model loading
│       ├── schemas.py       # Pydantic response models
│       └── input_handler.py # PDF→images, URL fetching
└── tests/
```

## Key Dependencies

- **pix2text** — OCR engine (text + math formula recognition + layout analysis)
- **fastapi** + **uvicorn** — Web framework + ASGI server
- **PyMuPDF (fitz)** — PDF to image conversion
- **httpx** — URL fetching
- **Pillow** — Image handling

## Constraints

- Python 3.10+
- Package managed with pyproject.toml and pip
- Models loaded once at startup, shared across requests
- All processing is local (no cloud API calls)
