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
