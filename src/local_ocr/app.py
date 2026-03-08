# src/local_ocr/app.py
import io
import tempfile
from contextlib import asynccontextmanager
from pathlib import Path
from urllib.parse import urlparse

from fastapi import FastAPI, File, UploadFile, HTTPException
from PIL import Image

from local_ocr.engine import OCREngine, extract_figures
from local_ocr.input_handler import pdf_to_images, open_pdf, pdf_page_to_image, fetch_url
from local_ocr.schemas import OCRResponse, Page, URLRequest

OUTPUT_DIR = Path("output")


def _save_output(
    pages: list[Page],
    stem: str,
    figures: list[tuple[str, Image.Image]] | None = None,
) -> Path:
    """Save combined page markdown and figure images to output/<stem>/."""
    out_dir = OUTPUT_DIR / stem
    out_dir.mkdir(parents=True, exist_ok=True)

    combined = "\n\n---\n\n".join(
        f"<!-- Page {p.page_number} -->\n\n{p.markdown}" for p in pages
    )
    out_path = out_dir / f"{stem}.md"
    out_path.write_text(combined, encoding="utf-8")

    for filename, img in figures or []:
        img.save(out_dir / filename)

    return out_path

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
    markdown, figures = extract_figures(markdown, image, page_num=1)
    pages = [Page(page_number=1, markdown=markdown)]
    stem = Path(file.filename or "image").stem
    out_path = _save_output(pages, stem, figures)
    return OCRResponse(pages=pages, output_path=str(out_path))


@app.post("/ocr/pdf", response_model=OCRResponse)
async def ocr_pdf(file: UploadFile = File(...)):
    contents = await file.read()
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=True) as tmp:
        tmp.write(contents)
        tmp.flush()
        doc = open_pdf(tmp.name)
        pages = []
        all_figures: list[tuple[str, Image.Image]] = []
        for i in range(len(doc)):
            pdf_page = doc[i]
            img = pdf_page_to_image(pdf_page)
            markdown = engine.recognize_image(img)
            markdown, figures = extract_figures(
                markdown, img, page_num=i + 1, pdf_page=pdf_page
            )
            pages.append(Page(page_number=i + 1, markdown=markdown))
            all_figures.extend(figures)
        doc.close()
    stem = Path(file.filename or "document").stem
    out_path = _save_output(pages, stem, all_figures)
    return OCRResponse(pages=pages, output_path=str(out_path))


@app.post("/ocr/url", response_model=OCRResponse)
async def ocr_url(request: URLRequest):
    try:
        result = fetch_url(request.url)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to fetch URL: {e}")

    parsed = urlparse(request.url)
    stem = Path(parsed.path).stem or "url_output"

    if result["type"] == "image":
        image = result["data"]
        markdown = engine.recognize_image(image)
        markdown, figures = extract_figures(markdown, image, page_num=1)
        pages = [Page(page_number=1, markdown=markdown)]
        out_path = _save_output(pages, stem, figures)
        return OCRResponse(pages=pages, output_path=str(out_path))

    # PDF
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=True) as tmp:
        tmp.write(result["data"])
        tmp.flush()
        doc = open_pdf(tmp.name)
        pages = []
        all_figures: list[tuple[str, Image.Image]] = []
        for i in range(len(doc)):
            pdf_page = doc[i]
            img = pdf_page_to_image(pdf_page)
            markdown = engine.recognize_image(img)
            markdown, figures = extract_figures(
                markdown, img, page_num=i + 1, pdf_page=pdf_page
            )
            pages.append(Page(page_number=i + 1, markdown=markdown))
            all_figures.extend(figures)
        doc.close()
    out_path = _save_output(pages, stem, all_figures)
    return OCRResponse(pages=pages, output_path=str(out_path))
