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
