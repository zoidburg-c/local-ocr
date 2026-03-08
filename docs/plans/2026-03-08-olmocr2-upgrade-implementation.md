# OlmOCR-2 Engine Upgrade Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Replace Pix2Text with OlmOCR-2 running via vLLM on an NVIDIA RTX 5090, outputting Markdown with LaTeX for math exam documents.

**Architecture:** FastAPI REST service with vLLM embedded engine loading `allenai/olmOCR-2-7B-1025` at startup. All inputs (images, PDFs, URLs) are normalized to PIL Images, passed to the VLM with the OlmOCR-2 prompt, and returned as per-page Markdown with LaTeX math. The YAML front matter from OlmOCR-2 output is stripped, returning clean Markdown.

**Tech Stack:** Python 3.10+, FastAPI, uvicorn, vLLM, torch (CUDA), PyMuPDF (fitz), httpx, Pillow

---

### Task 1: Update Dependencies

**Files:**
- Modify: `pyproject.toml`

**Step 1: Update pyproject.toml**

Replace `pix2text` dependency with `vllm` and `torch`. Remove `numpy` (no longer needed directly).

```toml
[build-system]
requires = ["setuptools>=68.0"]
build-backend = "setuptools.build_meta"

[project]
name = "local-ocr"
version = "0.2.0"
requires-python = ">=3.10"
dependencies = [
    "fastapi>=0.115",
    "uvicorn[standard]>=0.34",
    "vllm>=0.7",
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

**Step 2: Install updated dependencies**

Run: `pip install -e ".[dev]"`

**Step 3: Commit**

```bash
git add pyproject.toml
git commit -m "chore: replace pix2text with vllm in dependencies"
```

---

### Task 2: Simplify Response Schemas

**Files:**
- Modify: `src/local_ocr/schemas.py`
- Modify: `tests/test_schemas.py`

**Step 1: Write the failing test**

Replace `tests/test_schemas.py` entirely:

```python
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
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_schemas.py -v`
Expected: FAIL (`Page` still expects `elements`, not `markdown`)

**Step 3: Write implementation**

Replace `src/local_ocr/schemas.py` entirely:

```python
# src/local_ocr/schemas.py
from pydantic import BaseModel


class Page(BaseModel):
    page_number: int
    markdown: str


class OCRResponse(BaseModel):
    pages: list[Page]


class URLRequest(BaseModel):
    url: str
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_schemas.py -v`
Expected: PASS (3 tests)

**Step 5: Commit**

```bash
git add src/local_ocr/schemas.py tests/test_schemas.py
git commit -m "feat: simplify schemas to markdown-based output"
```

---

### Task 3: Rewrite OCR Engine with vLLM

**Files:**
- Modify: `src/local_ocr/engine.py`
- Modify: `tests/test_engine.py`

**Step 1: Write the failing test**

Replace `tests/test_engine.py` entirely:

```python
# tests/test_engine.py
from unittest.mock import MagicMock, patch
from PIL import Image

from local_ocr.engine import OCREngine, OLMOCR_PROMPT, _strip_yaml_front_matter


def test_strip_yaml_front_matter():
    text = (
        '---\nprimary_language: "en"\nis_table: false\n---\n'
        "# Title\n\nSome content with $x^2$"
    )
    result = _strip_yaml_front_matter(text)
    assert result == "# Title\n\nSome content with $x^2$"


def test_strip_yaml_front_matter_no_front_matter():
    text = "# Title\n\nNo front matter here"
    result = _strip_yaml_front_matter(text)
    assert result == text


@patch("local_ocr.engine.LLM")
def test_recognize_image(mock_llm_cls):
    mock_output = MagicMock()
    mock_output.outputs = [MagicMock(text="# Question 1\n\nSolve $x^2 = 4$")]
    mock_instance = MagicMock()
    mock_instance.generate.return_value = [mock_output]
    mock_llm_cls.return_value = mock_instance

    engine = OCREngine()
    image = Image.new("RGB", (100, 100), "white")
    result = engine.recognize_image(image)

    assert "# Question 1" in result
    assert "$x^2 = 4$" in result
    mock_instance.generate.assert_called_once()


@patch("local_ocr.engine.LLM")
def test_recognize_image_strips_front_matter(mock_llm_cls):
    raw_output = (
        '---\nprimary_language: "en"\nis_table: false\n---\n'
        "# Math Exam\n\n$$\\int_0^1 x^2 dx$$"
    )
    mock_output = MagicMock()
    mock_output.outputs = [MagicMock(text=raw_output)]
    mock_instance = MagicMock()
    mock_instance.generate.return_value = [mock_output]
    mock_llm_cls.return_value = mock_instance

    engine = OCREngine()
    image = Image.new("RGB", (100, 100), "white")
    result = engine.recognize_image(image)

    assert result.startswith("# Math Exam")
    assert "---" not in result


@patch("local_ocr.engine.LLM")
def test_recognize_image_resizes_large_image(mock_llm_cls):
    mock_output = MagicMock()
    mock_output.outputs = [MagicMock(text="content")]
    mock_instance = MagicMock()
    mock_instance.generate.return_value = [mock_output]
    mock_llm_cls.return_value = mock_instance

    engine = OCREngine()
    large_image = Image.new("RGB", (4000, 3000), "white")
    engine.recognize_image(large_image)

    # Verify the image passed to generate was resized
    call_args = mock_instance.generate.call_args
    inputs = call_args[0][0]
    passed_image = inputs[0]["multi_modal_data"]["image"]
    assert max(passed_image.size) <= 1288


@patch("local_ocr.engine.LLM")
def test_prompt_contains_latex_instruction(mock_llm_cls):
    mock_output = MagicMock()
    mock_output.outputs = [MagicMock(text="content")]
    mock_instance = MagicMock()
    mock_instance.generate.return_value = [mock_output]
    mock_llm_cls.return_value = mock_instance

    engine = OCREngine()
    image = Image.new("RGB", (100, 100), "white")
    engine.recognize_image(image)

    call_args = mock_instance.generate.call_args
    inputs = call_args[0][0]
    prompt = inputs[0]["prompt"]
    assert "LateX" in prompt or "LaTeX" in prompt
    assert "<|vision_start|>" in prompt
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_engine.py -v`
Expected: FAIL (old engine code, missing new imports)

**Step 3: Write implementation**

Replace `src/local_ocr/engine.py` entirely:

```python
# src/local_ocr/engine.py
import re

from PIL import Image
from vllm import LLM, SamplingParams

_MAX_IMAGE_DIM = 1288

OLMOCR_PROMPT = (
    "Attached is one page of a document that you must process. "
    "Just return the plain text representation of this document "
    "as if you were reading it naturally. Convert equations to LateX "
    "and tables to HTML.\n"
    "If there are any figures or charts, label them with the following "
    "markdown syntax ![Alt text describing the contents of the "
    "figure](page_startx_starty_width_height.png)\n"
    "Return your output as markdown, with a front matter section on top "
    "specifying values for the primary_language, is_rotation_valid, "
    "rotation_correction, is_table, and is_diagram parameters."
)

_CHAT_PROMPT = (
    "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
    "<|im_start|>user\n"
    "<|vision_start|><|image_pad|><|vision_end|>"
    f"{OLMOCR_PROMPT}<|im_end|>\n"
    "<|im_start|>assistant\n"
)

_FRONT_MATTER_RE = re.compile(r"^---\s*\n.*?\n---\s*\n", re.DOTALL)


def _strip_yaml_front_matter(text: str) -> str:
    """Remove YAML front matter block from OlmOCR-2 output."""
    return _FRONT_MATTER_RE.sub("", text)


def _resize_for_inference(image: Image.Image) -> Image.Image:
    """Resize image so longest dimension <= _MAX_IMAGE_DIM."""
    max_dim = max(image.size)
    if max_dim <= _MAX_IMAGE_DIM:
        return image
    scale = _MAX_IMAGE_DIM / max_dim
    new_size = (int(image.width * scale), int(image.height * scale))
    return image.resize(new_size, Image.LANCZOS)


class OCREngine:
    def __init__(
        self,
        model_name: str = "allenai/olmOCR-2-7B-1025",
        max_model_len: int = 16384,
    ) -> None:
        self._llm = LLM(
            model=model_name,
            max_model_len=max_model_len,
            max_num_seqs=5,
            mm_processor_kwargs={
                "min_pixels": 28 * 28,
                "max_pixels": 1280 * 28 * 28,
            },
        )
        self._sampling_params = SamplingParams(
            temperature=0.1,
            max_tokens=8000,
        )

    def recognize_image(self, image: Image.Image) -> str:
        """Recognize text and formulas in an image. Returns Markdown with LaTeX."""
        image = _resize_for_inference(image)
        outputs = self._llm.generate(
            [{"prompt": _CHAT_PROMPT, "multi_modal_data": {"image": image}}],
            sampling_params=self._sampling_params,
        )
        raw_text = outputs[0].outputs[0].text
        return _strip_yaml_front_matter(raw_text).strip()
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_engine.py -v`
Expected: PASS (6 tests)

**Step 5: Commit**

```bash
git add src/local_ocr/engine.py tests/test_engine.py
git commit -m "feat: rewrite engine with vLLM and OlmOCR-2"
```

---

### Task 4: Update FastAPI App for Markdown Output

**Files:**
- Modify: `src/local_ocr/app.py`
- Modify: `tests/test_app.py`

**Step 1: Write the failing test**

Replace `tests/test_app.py` entirely:

```python
# tests/test_app.py
from unittest.mock import patch, MagicMock
import pytest
from httpx import AsyncClient, ASGITransport
from PIL import Image
import io


def _make_test_image_bytes() -> bytes:
    img = Image.new("RGB", (100, 100), "white")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


MOCK_MARKDOWN = "# Question 1\n\nSolve $x^2 = 4$\n\n$$x = \\pm 2$$"


@pytest.fixture
def mock_engine():
    with patch("local_ocr.app.engine") as mock:
        mock.recognize_image.return_value = MOCK_MARKDOWN
        yield mock


async def test_health():
    from local_ocr.app import app

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"


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
    assert data["pages"][0]["page_number"] == 1
    assert "$x^2 = 4$" in data["pages"][0]["markdown"]


async def test_ocr_pdf(mock_engine):
    from local_ocr.app import app

    with patch("local_ocr.app.pdf_to_images") as mock_pdf:
        mock_pdf.return_value = [
            Image.new("RGB", (100, 100), "white"),
            Image.new("RGB", (100, 100), "white"),
        ]
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post(
                "/ocr/pdf",
                files={"file": ("test.pdf", b"%PDF-fake", "application/pdf")},
            )
    assert resp.status_code == 200
    data = resp.json()
    assert len(data["pages"]) == 2
    assert data["pages"][0]["page_number"] == 1
    assert data["pages"][1]["page_number"] == 2
    assert "$x^2 = 4$" in data["pages"][0]["markdown"]


async def test_ocr_url_image(mock_engine):
    from local_ocr.app import app

    with patch("local_ocr.app.fetch_url") as mock_fetch:
        mock_fetch.return_value = {
            "type": "image",
            "data": Image.new("RGB", (100, 100), "white"),
        }
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post(
                "/ocr/url", json={"url": "http://example.com/img.png"}
            )
    assert resp.status_code == 200
    data = resp.json()
    assert len(data["pages"]) == 1
    assert "$x^2 = 4$" in data["pages"][0]["markdown"]


async def test_ocr_url_pdf(mock_engine):
    from local_ocr.app import app

    with patch("local_ocr.app.fetch_url") as mock_fetch:
        mock_fetch.return_value = {"type": "pdf", "data": b"%PDF-fake"}

        with patch("local_ocr.app.pdf_to_images") as mock_pdf:
            mock_pdf.return_value = [Image.new("RGB", (100, 100), "white")]
            transport = ASGITransport(app=app)
            async with AsyncClient(
                transport=transport, base_url="http://test"
            ) as client:
                resp = await client.post(
                    "/ocr/url", json={"url": "http://example.com/doc.pdf"}
                )
    assert resp.status_code == 200
    data = resp.json()
    assert len(data["pages"]) == 1


async def test_ocr_url_bad_url():
    from local_ocr.app import app

    with patch("local_ocr.app.engine"):
        with patch("local_ocr.app.fetch_url", side_effect=Exception("Connection error")):
            transport = ASGITransport(app=app)
            async with AsyncClient(
                transport=transport, base_url="http://test"
            ) as client:
                resp = await client.post(
                    "/ocr/url", json={"url": "http://bad-url.example.com"}
                )
    assert resp.status_code == 400
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_app.py -v`
Expected: FAIL (app still returns old element-based schema)

**Step 3: Write implementation**

Replace `src/local_ocr/app.py` entirely:

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


app = FastAPI(title="Local OCR", version="0.2.0", lifespan=lifespan)


@app.get("/health")
async def health():
    return {"status": "ok", "model_loaded": engine is not None}


@app.post("/ocr/image", response_model=OCRResponse)
async def ocr_image(file: UploadFile = File(...)):
    contents = await file.read()
    image = Image.open(io.BytesIO(contents))
    markdown = engine.recognize_image(image)
    return OCRResponse(pages=[Page(page_number=1, markdown=markdown)])


@app.post("/ocr/pdf", response_model=OCRResponse)
async def ocr_pdf(file: UploadFile = File(...)):
    contents = await file.read()
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=True) as tmp:
        tmp.write(contents)
        tmp.flush()
        images = pdf_to_images(tmp.name)

    pages = []
    for i, img in enumerate(images, start=1):
        markdown = engine.recognize_image(img)
        pages.append(Page(page_number=i, markdown=markdown))
    return OCRResponse(pages=pages)


@app.post("/ocr/url", response_model=OCRResponse)
async def ocr_url(request: URLRequest):
    try:
        result = fetch_url(request.url)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to fetch URL: {e}")

    if result["type"] == "image":
        markdown = engine.recognize_image(result["data"])
        return OCRResponse(pages=[Page(page_number=1, markdown=markdown)])

    # PDF
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=True) as tmp:
        tmp.write(result["data"])
        tmp.flush()
        images = pdf_to_images(tmp.name)

    pages = []
    for i, img in enumerate(images, start=1):
        markdown = engine.recognize_image(img)
        pages.append(Page(page_number=i, markdown=markdown))
    return OCRResponse(pages=pages)
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_app.py -v`
Expected: PASS (6 tests)

**Step 5: Commit**

```bash
git add src/local_ocr/app.py tests/test_app.py
git commit -m "feat: update app endpoints for markdown output"
```

---

### Task 5: Update CLAUDE.md

**Files:**
- Modify: `CLAUDE.md`

**Step 1: Update CLAUDE.md**

```markdown
# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build and Run

```bash
# Install in dev mode
pip install -e ".[dev]"

# Run the server (requires NVIDIA GPU with CUDA)
uvicorn local_ocr.app:app --reload --host 0.0.0.0 --port 8000

# Run all tests
pytest -v

# Run a single test
pytest tests/test_engine.py::test_recognize_image -v
```

## Architecture

FastAPI REST service wrapping OlmOCR-2 (7B VLM) via vLLM for local OCR of text, math equations (LaTeX), and diagrams. Outputs Markdown with embedded LaTeX.

- `src/local_ocr/app.py` — FastAPI endpoints: `POST /ocr/image`, `POST /ocr/pdf`, `POST /ocr/url`, `GET /health`
- `src/local_ocr/engine.py` — OlmOCR-2 wrapper via vLLM. Loads model once at startup via FastAPI lifespan. Constructs Qwen2.5-VL chat prompt with OlmOCR-2's document processing instruction. Strips YAML front matter from output, returns clean Markdown with LaTeX.
- `src/local_ocr/input_handler.py` — PDF-to-PIL-images conversion (PyMuPDF), URL fetching (httpx)
- `src/local_ocr/schemas.py` — Pydantic models: `Page` (with `markdown` field), `OCRResponse`, `URLRequest`

The `OCREngine` is a module-level singleton initialized during FastAPI lifespan. All inputs (images, PDFs, URLs) are normalized to PIL Images before passing to the engine. Images are resized so the longest dimension is <= 1288px before inference.

## OlmOCR-2 Integration

- Model: `allenai/olmOCR-2-7B-1025` (fine-tuned Qwen2.5-VL-7B)
- Inference: vLLM embedded engine with FP16, `max_model_len=16384`
- Prompt: Uses Qwen2.5-VL chat template tokens (`<|im_start|>`, `<|vision_start|>`, etc.) with OlmOCR-2's document processing prompt
- Output: Markdown with YAML front matter (stripped before returning). LaTeX math in `$...$` (inline) and `$$...$$` (block) delimiters.

## Testing

Tests mock `vllm.LLM` — no actual model loading or GPU needed during tests. The FastAPI app tests use `httpx.AsyncClient` with `ASGITransport`.

- `asyncio_mode = "auto"` is set in `pyproject.toml`, so async test functions are detected automatically (no `@pytest.mark.asyncio` needed).
- App tests mock the module-level `engine` global via `patch("local_ocr.app.engine")` rather than patching `vllm.LLM` directly.
```

**Step 2: Commit**

```bash
git add CLAUDE.md
git commit -m "docs: update CLAUDE.md for OlmOCR-2 engine"
```

---

### Task 6: Run Full Test Suite

**Step 1: Run all tests**

Run: `pytest -v`
Expected: All tests pass (3 schema + 6 engine + 6 app + 4 input_handler = 19 tests).

**Step 2: Fix any failures**

If any tests fail, fix them and commit:

```bash
git add -A
git commit -m "fix: resolve test failures from engine migration"
```
