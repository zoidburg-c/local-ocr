# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build and Run

```bash
# Install in dev mode
pip install -e ".[dev]"

# Run the server
uvicorn local_ocr.app:app --reload --host 0.0.0.0 --port 8000

# Run all tests
pytest -v

# Run a single test
pytest tests/test_engine.py::test_recognize_text -v
```

## Architecture

FastAPI REST service wrapping Pix2Text for local OCR of text, math equations (LaTeX), and diagrams.

- `src/local_ocr/app.py` — FastAPI endpoints: `POST /ocr/image`, `POST /ocr/pdf`, `POST /ocr/url`, `GET /health`
- `src/local_ocr/engine.py` — Pix2Text wrapper. Loads model once at startup via FastAPI lifespan. Maps P2T types (`text`, `isolated`, `embedding`) to our schema types (`text`, `formula`).
- `src/local_ocr/input_handler.py` — PDF-to-PIL-images conversion (PyMuPDF), URL fetching (httpx)
- `src/local_ocr/schemas.py` — Pydantic models: `Element`, `Page`, `OCRResponse`, `URLRequest`

The `OCREngine` is a module-level singleton initialized during FastAPI lifespan. All inputs (images, PDFs, URLs) are normalized to PIL Images before passing to the engine.

## Pix2Text Type Mapping

P2T `recognize_text_formula(img, return_text=False)` returns dicts with `type` as `"text"`, `"isolated"` (standalone formula), or `"embedding"` (inline formula). We map both formula types to `"formula"` and populate the `latex` field. Position arrays (4 corner points) are converted to `[x_min, y_min, width, height]` bounding boxes.

## Testing

Tests mock `Pix2Text` and `httpx` — no actual model loading or network calls during tests. The FastAPI app tests use `httpx.AsyncClient` with `ASGITransport`.
