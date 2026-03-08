# OlmOCR-2 Engine Upgrade — Design

## Purpose

Replace Pix2Text with OlmOCR-2 (7B fine-tuned Qwen2.5-VL) to achieve best-in-class OCR for math exams across all levels (K-12 through graduate), including handwritten content. Output changes from element-level JSON to page-level Markdown with embedded LaTeX. Leverages NVIDIA RTX 5090 (32GB VRAM) for local GPU inference.

## Why OlmOCR-2

- 82.4% on olmOCR-Bench, RL-fine-tuned specifically for math equations and tables
- Handles both printed and handwritten math documents
- Outputs structured Markdown with LaTeX natively
- ~14-17 GB VRAM in FP16 — fits comfortably on 32GB 5090
- Fully Apache 2.0 (code, data, weights — no restrictions)
- Built on Qwen2.5-VL-7B-Instruct, actively maintained by Allen AI

Alternatives considered: Chandra OCR (better benchmarks but restrictive weight license), Marker + VLM multi-engine (more complexity, GPL license), Pix2Text (current — quality ceiling too low).

## Architecture

```
Client → FastAPI → Input Handler (image/PDF/URL) → OlmOCR-2 via vLLM (GPU) → Markdown+LaTeX response
```

Same REST API surface. Same input normalization (all inputs become PIL Images). The engine and response format change.

### Engine (vLLM embedded)

vLLM's `LLM` class runs inside the FastAPI process, loaded at startup via lifespan. Single process deployment.

- Model: `allenai/olmOCR-2-7B-1025`
- Precision: FP16 on CUDA
- Inference: `LLM.generate()` with `SamplingParams(max_tokens=4096, temperature=0.0)`
- Prompt: Instructs the VLM to output document content as Markdown with LaTeX math delimiters (`$...$` inline, `$$...$$` block)

### Response Schema (simplified)

```json
{
  "pages": [
    {
      "page_number": 1,
      "markdown": "## Question 1\n\nFind the derivative of $f(x) = x^3 + 2x$\n\n$$f'(x) = 3x^2 + 2$$"
    }
  ]
}
```

Old element-based schema (`Element`, `position`, `score`) is removed entirely. Clean replacement, no backward compatibility.

## Components Changed

| File | Change |
|------|--------|
| `engine.py` | Complete rewrite. vLLM-based OlmOCR-2 inference. |
| `schemas.py` | Simplified. `Page.markdown: str` replaces `Page.elements`. `Element` class removed. |
| `app.py` | Endpoints return Markdown content per page. Health reports GPU/model status. |
| `input_handler.py` | No changes. |
| `pyproject.toml` | Replace `pix2text` with `vllm`, `torch` (CUDA), `transformers`. |

## Dependencies

Remove: `pix2text`, `numpy` (no longer needed for bbox conversion)

Add: `vllm`, `torch` (CUDA build), `transformers`

Keep: `fastapi`, `uvicorn`, `PyMuPDF`, `httpx`, `Pillow`, `python-multipart`

## Testing

Same strategy — mock `vllm.LLM` in tests. No GPU or model loading in CI.

- Verify prompt construction for single images
- Verify Markdown passthrough to response schema
- Multi-page PDFs produce per-page Markdown
- URL endpoint handles image and PDF content types
- Health endpoint reports model status

## Constraints

- Python 3.10+
- NVIDIA GPU required (CUDA). CPU fallback not supported.
- Model downloaded from HuggingFace on first run (~14GB)
- All processing remains local (no cloud API calls)
