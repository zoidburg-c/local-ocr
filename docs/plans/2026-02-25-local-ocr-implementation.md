# Local OCR Service Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build a FastAPI REST service that uses Pix2Text to extract text, math equations (LaTeX), and diagrams from images, PDFs, and URLs.

**Architecture:** FastAPI app with a singleton Pix2Text engine loaded at startup. Three endpoints accept images, PDFs, or URLs. Input handler normalizes all inputs to PIL Images, which the engine processes. Results are returned as structured JSON with element types (text, formula, diagram) and positions.

**Tech Stack:** Python 3.10+, FastAPI, uvicorn, pix2text, PyMuPDF (fitz), httpx, Pillow

---

### Task 1: Project Scaffolding

**Files:**
- Create: `pyproject.toml`
- Create: `src/local_ocr/__init__.py`

**Step 1: Create pyproject.toml**

```toml
[build-system]
requires = ["setuptools>=68.0"]
build-backend = "setuptools.backends._legacy:_Backend"

[project]
name = "local-ocr"
version = "0.1.0"
requires-python = ">=3.10"
dependencies = [
    "fastapi>=0.115",
    "uvicorn[standard]>=0.34",
    "pix2text>=1.1",
    "PyMuPDF>=1.25",
    "httpx>=0.28",
    "Pillow>=11.0",
    "python-multipart>=0.0.18",
]

[project.optional-dependencies]
dev = [
    "pytest>=8.0",
    "pytest-asyncio>=0.25",
    "httpx",
]

[tool.setuptools.packages.find]
where = ["src"]

[tool.pytest.ini_options]
asyncio_mode = "auto"
testpaths = ["tests"]
```

**Step 2: Create package init**

```python
# src/local_ocr/__init__.py
```

(Empty file, just marks the package.)

**Step 3: Create directory structure**

```bash
mkdir -p src/local_ocr tests
```

**Step 4: Install in dev mode**

```bash
pip install -e ".[dev]"
```

**Step 5: Commit**

```bash
git add pyproject.toml src/local_ocr/__init__.py
git commit -m "feat: project scaffolding with pyproject.toml"
```

---

### Task 2: Pydantic Response Schemas

**Files:**
- Create: `src/local_ocr/schemas.py`
- Create: `tests/test_schemas.py`

**Step 1: Write the failing test**

```python
# tests/test_schemas.py
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
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_schemas.py -v`
Expected: FAIL (ImportError)

**Step 3: Write implementation**

```python
# src/local_ocr/schemas.py
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
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_schemas.py -v`
Expected: PASS (4 tests)

**Step 5: Commit**

```bash
git add src/local_ocr/schemas.py tests/test_schemas.py
git commit -m "feat: add Pydantic response schemas"
```

---

### Task 3: Pix2Text Engine Wrapper

**Files:**
- Create: `src/local_ocr/engine.py`
- Create: `tests/test_engine.py`

The engine wraps Pix2Text. It loads the model once and provides a `recognize(image) -> list[Element]` method that normalizes Pix2Text output into our schema.

Pix2Text `recognize_text_formula(img, return_text=False)` returns a list of dicts:
```python
{"type": "text"|"isolated"|"embedding", "text": "...", "score": 0.95, "position": np.ndarray([4,2]), "line_number": 0}
```
- `"isolated"` = standalone formula → map to `"formula"`
- `"embedding"` = inline formula → map to `"formula"`
- `"text"` = plain text → map to `"text"`

Position is a numpy array of 4 corner points `[[x1,y1],[x2,y2],[x3,y3],[x4,y4]]`. We convert to `[x_min, y_min, width, height]`.

**Step 1: Write the failing test**

```python
# tests/test_engine.py
from unittest.mock import MagicMock, patch
import numpy as np
from local_ocr.engine import OCREngine


def _make_p2t_result(type_: str, text: str, score: float = 0.9):
    return {
        "type": type_,
        "text": text,
        "score": score,
        "position": np.array([[10, 20], [110, 20], [110, 70], [10, 70]]),
        "line_number": 0,
    }


@patch("local_ocr.engine.Pix2Text")
def test_recognize_text(mock_p2t_cls):
    mock_instance = MagicMock()
    mock_instance.recognize_text_formula.return_value = [
        _make_p2t_result("text", "hello world"),
    ]
    mock_p2t_cls.return_value = mock_instance

    engine = OCREngine()
    elements = engine.recognize_image("fake_path.png")

    assert len(elements) == 1
    assert elements[0].type == "text"
    assert elements[0].content == "hello world"
    assert elements[0].position == [10.0, 20.0, 100.0, 50.0]


@patch("local_ocr.engine.Pix2Text")
def test_recognize_formula(mock_p2t_cls):
    mock_instance = MagicMock()
    mock_instance.recognize_text_formula.return_value = [
        _make_p2t_result("isolated", "E = mc^{2}"),
    ]
    mock_p2t_cls.return_value = mock_instance

    engine = OCREngine()
    elements = engine.recognize_image("fake_path.png")

    assert len(elements) == 1
    assert elements[0].type == "formula"
    assert elements[0].latex == "E = mc^{2}"


@patch("local_ocr.engine.Pix2Text")
def test_recognize_embedding_formula(mock_p2t_cls):
    mock_instance = MagicMock()
    mock_instance.recognize_text_formula.return_value = [
        _make_p2t_result("embedding", "x^2"),
    ]
    mock_p2t_cls.return_value = mock_instance

    engine = OCREngine()
    elements = engine.recognize_image("fake_path.png")

    assert elements[0].type == "formula"
    assert elements[0].latex == "x^2"
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_engine.py -v`
Expected: FAIL (ImportError)

**Step 3: Write implementation**

```python
# src/local_ocr/engine.py
from pathlib import Path

import numpy as np
from PIL import Image
from pix2text import Pix2Text

from local_ocr.schemas import Element

# Map Pix2Text types to our schema types
_TYPE_MAP = {
    "text": "text",
    "isolated": "formula",
    "embedding": "formula",
}


def _position_to_bbox(position: np.ndarray) -> list[float]:
    """Convert 4-corner points [[x,y],...] to [x_min, y_min, width, height]."""
    x_min = float(position[:, 0].min())
    y_min = float(position[:, 1].min())
    x_max = float(position[:, 0].max())
    y_max = float(position[:, 1].max())
    return [x_min, y_min, x_max - x_min, y_max - y_min]


class OCREngine:
    def __init__(self) -> None:
        self._p2t = Pix2Text()

    def recognize_image(self, image: str | Path | Image.Image) -> list[Element]:
        """Recognize text and formulas in an image. Returns list of Elements."""
        results = self._p2t.recognize_text_formula(image, return_text=False)
        elements = []
        for item in results:
            elem_type = _TYPE_MAP.get(item["type"], "text")
            text = item.get("text", "")
            elements.append(
                Element(
                    type=elem_type,
                    content=text,
                    latex=text if elem_type == "formula" else None,
                    position=_position_to_bbox(item["position"]),
                    score=item.get("score"),
                )
            )
        return elements
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_engine.py -v`
Expected: PASS (3 tests)

**Step 5: Commit**

```bash
git add src/local_ocr/engine.py tests/test_engine.py
git commit -m "feat: add Pix2Text engine wrapper"
```

---

### Task 4: Input Handler (PDF + URL)

**Files:**
- Create: `src/local_ocr/input_handler.py`
- Create: `tests/test_input_handler.py`

The input handler converts PDFs to PIL Images (via PyMuPDF) and fetches URLs (via httpx).

**Step 1: Write the failing test**

```python
# tests/test_input_handler.py
from unittest.mock import patch, MagicMock
from PIL import Image
from local_ocr.input_handler import pdf_to_images, fetch_url


def test_pdf_to_images_returns_pil_images(tmp_path):
    """Use a tiny valid PDF to test conversion."""
    # Create a minimal 1-page PDF with PyMuPDF
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
    # Create a small PNG in memory
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
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_input_handler.py -v`
Expected: FAIL (ImportError)

**Step 3: Write implementation**

```python
# src/local_ocr/input_handler.py
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
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_input_handler.py -v`
Expected: PASS (4 tests)

**Step 5: Commit**

```bash
git add src/local_ocr/input_handler.py tests/test_input_handler.py
git commit -m "feat: add input handler for PDF and URL processing"
```

---

### Task 5: FastAPI Application and Endpoints

**Files:**
- Create: `src/local_ocr/app.py`
- Create: `tests/test_app.py`

**Step 1: Write the failing test**

```python
# tests/test_app.py
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
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_app.py -v`
Expected: FAIL (ImportError)

**Step 3: Write implementation**

```python
# src/local_ocr/app.py
import io
import tempfile
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, UploadFile, HTTPException
from PIL import Image

from local_ocr.engine import OCREngine
from local_ocr.input_handler import pdf_to_images, fetch_url
from local_ocr.schemas import OCRResponse, Page, URLRequest

engine: OCREngine = None  # type: ignore[assignment]


@asynccontextmanager
async def lifespan(app: FastAPI):
    global engine
    engine = OCREngine()
    yield


app = FastAPI(title="Local OCR", version="0.1.0", lifespan=lifespan)


@app.get("/health")
async def health():
    return {"status": "ok", "model_loaded": engine is not None}


@app.post("/ocr/image", response_model=OCRResponse)
async def ocr_image(file: UploadFile = File(...)):
    contents = await file.read()
    image = Image.open(io.BytesIO(contents))
    elements = engine.recognize_image(image)
    return OCRResponse(pages=[Page(page_number=1, elements=elements)])


@app.post("/ocr/pdf", response_model=OCRResponse)
async def ocr_pdf(file: UploadFile = File(...)):
    contents = await file.read()
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=True) as tmp:
        tmp.write(contents)
        tmp.flush()
        images = pdf_to_images(tmp.name)

    pages = []
    for i, img in enumerate(images, start=1):
        elements = engine.recognize_image(img)
        pages.append(Page(page_number=i, elements=elements))
    return OCRResponse(pages=pages)


@app.post("/ocr/url", response_model=OCRResponse)
async def ocr_url(request: URLRequest):
    try:
        result = fetch_url(request.url)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to fetch URL: {e}")

    if result["type"] == "image":
        elements = engine.recognize_image(result["data"])
        return OCRResponse(pages=[Page(page_number=1, elements=elements)])

    # PDF
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=True) as tmp:
        tmp.write(result["data"])
        tmp.flush()
        images = pdf_to_images(tmp.name)

    pages = []
    for i, img in enumerate(images, start=1):
        elements = engine.recognize_image(img)
        pages.append(Page(page_number=i, elements=elements))
    return OCRResponse(pages=pages)
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_app.py -v`
Expected: PASS (4 tests)

**Step 5: Commit**

```bash
git add src/local_ocr/app.py tests/test_app.py
git commit -m "feat: add FastAPI app with image, PDF, and URL endpoints"
```

---

### Task 6: Entry Point and CLAUDE.md

**Files:**
- Create: `CLAUDE.md`

**Step 1: Create CLAUDE.md**

```markdown
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

- `src/local_ocr/app.py` — FastAPI endpoints: `/ocr/image`, `/ocr/pdf`, `/ocr/url`, `/health`
- `src/local_ocr/engine.py` — Pix2Text wrapper. Loads model at startup. Maps P2T types (`text`, `isolated`, `embedding`) to our schema (`text`, `formula`).
- `src/local_ocr/input_handler.py` — PDF→PIL images (PyMuPDF), URL fetching (httpx)
- `src/local_ocr/schemas.py` — Pydantic models: `Element`, `Page`, `OCRResponse`, `URLRequest`

The engine is a module-level singleton initialized during FastAPI lifespan. All inputs are normalized to PIL Images before passing to the engine.

## Pix2Text Type Mapping

P2T returns `type` as `"text"`, `"isolated"` (standalone formula), or `"embedding"` (inline formula). We map both formula types to `"formula"` and populate the `latex` field.
```

**Step 2: Commit**

```bash
git add CLAUDE.md
git commit -m "docs: add CLAUDE.md"
```

---

### Task 7: Run Full Test Suite

**Step 1: Run all tests**

Run: `pytest -v`
Expected: All 11 tests pass.

**Step 2: Start the server and smoke test**

Run: `uvicorn local_ocr.app:app --host 0.0.0.0 --port 8000 &`
Run: `curl http://localhost:8000/health`
Expected: `{"status":"ok","model_loaded":true}`

**Step 3: Final commit (if any fixes needed)**

```bash
git add -A
git commit -m "fix: any adjustments from integration testing"
```
