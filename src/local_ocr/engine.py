# src/local_ocr/engine.py
import re

import torch
from PIL import Image
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor

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

_FRONT_MATTER_RE = re.compile(r"^---\s*\n.*?\n---\s*\n?", re.DOTALL)
_FIGURE_RE = re.compile(
    r"!\[([^\]]*)\]\(page_(\d+)_(\d+)_(\d+)_(\d+)\.png\)"
)


def _strip_yaml_front_matter(text: str) -> str:
    """Remove YAML front matter block from OlmOCR-2 output."""
    return _FRONT_MATTER_RE.sub("", text)


def _normalize_latex_delimiters(text: str) -> str:
    """Normalize LaTeX delimiters for broad renderer compatibility.

    - Converts \\( \\) → $...$  and  \\[ \\] → $$...$$
    - Strips whitespace inside inline $...$ so strict renderers
      (Obsidian, some MathJax configs) render correctly.
    """
    # Convert bracket-style to dollar-style
    text = re.sub(r"\\\(", "$", text)
    text = re.sub(r"\\\)", "$", text)
    text = re.sub(r"\\\[", "$$", text)
    text = re.sub(r"\\\]", "$$", text)
    # Strip whitespace inside inline $...$ (but not display $$...$$)
    text = re.sub(
        r"(?<!\$)\$([^$]+)\$(?!\$)",
        lambda m: f"${m.group(1).strip()}$",
        text,
    )
    # Wrap display blocks that use \\ for line breaks in aligned environment
    text = _wrap_multiline_display_math(text)
    return text


def _wrap_multiline_display_math(text: str) -> str:
    r"""Wrap $$ blocks containing \\ line breaks in \begin{aligned}...\end{aligned}.

    Converts:
        $$
        eq1 \\
        eq2
        $$
    To:
        $$
        \begin{aligned}
        eq1 \\
        eq2
        \end{aligned}
        $$

    Skips blocks that already use a LaTeX environment (\begin{).
    """
    def _wrap(match: re.Match) -> str:
        content = match.group(1)
        # Skip if already in an environment (cases, aligned, etc.)
        if r"\begin{" in content:
            return match.group(0)
        # Only wrap if there are \\ line breaks
        if r"\\" not in content:
            return match.group(0)
        return f"$$\n\\begin{{aligned}}\n{content.strip()}\n\\end{{aligned}}\n$$"

    return re.sub(r"\$\$\n(.*?)\n\$\$", _wrap, text, flags=re.DOTALL)


def _inference_size(original: tuple[int, int]) -> tuple[int, int]:
    """Return the image dimensions after _resize_for_inference."""
    w, h = original
    max_dim = max(w, h)
    if max_dim <= _MAX_IMAGE_DIM:
        return (w, h)
    scale = _MAX_IMAGE_DIM / max_dim
    return (int(w * scale), int(h * scale))


_FIGURE_PADDING_PTS = 15  # padding around detected figure regions in PDF points
_RASTER_PAD_FRAC = 0.30   # expand model's estimate by 30% on each side for raster crops
_MIN_GRAPHIC_SIZE = 15  # ignore images/drawings smaller than this (pixels or pts)
_MAX_PAGE_COVERAGE = 0.6  # ignore elements covering more than 60% of page area


def _detect_figure_regions(pdf_page) -> list:
    """Detect figure regions on a PDF page by clustering graphical elements.

    Skips full-page images (scanned pages) and elements that cover most of the
    page.  Returns a list of fitz.Rect bounding boxes, one per detected figure.
    """
    import fitz as _fitz

    rects: list = []
    page_area = pdf_page.rect.width * pdf_page.rect.height

    # Collect image rects (skip tiny and full-page images)
    doc = pdf_page.parent
    for img_info in pdf_page.get_images(full=True):
        xref = img_info[0]
        pix = _fitz.Pixmap(doc, xref)
        if pix.width < _MIN_GRAPHIC_SIZE or pix.height < _MIN_GRAPHIC_SIZE:
            continue
        for r in pdf_page.get_image_rects(xref):
            if r.is_empty or r.is_infinite:
                continue
            # Skip images that cover most of the page (scanned pages)
            if (r.width * r.height) / page_area > _MAX_PAGE_COVERAGE:
                continue
            rects.append(r)

    # Collect drawing rects (skip page-spanning elements like borders)
    for d in pdf_page.get_drawings():
        r = d["rect"]
        if r.is_empty or r.is_infinite:
            continue
        if r.width < _MIN_GRAPHIC_SIZE and r.height < _MIN_GRAPHIC_SIZE:
            continue
        if (r.width * r.height) / page_area > _MAX_PAGE_COVERAGE:
            continue
        rects.append(r)

    if not rects:
        return []

    # Cluster overlapping/nearby rects into figure regions.
    # Two rects belong to the same figure if they overlap or are within
    # _FIGURE_PADDING_PTS of each other.
    pad = _FIGURE_PADDING_PTS
    clusters: list[_fitz.Rect] = []
    for r in rects:
        expanded = _fitz.Rect(r.x0 - pad, r.y0 - pad, r.x1 + pad, r.y1 + pad)
        merged = False
        for i, c in enumerate(clusters):
            if c.intersects(expanded):
                clusters[i] = c | r  # union
                merged = True
                break
        if not merged:
            clusters.append(_fitz.Rect(r))

    # Second pass: merge clusters that now overlap after growth
    changed = True
    while changed:
        changed = False
        new_clusters: list[_fitz.Rect] = []
        for c in clusters:
            expanded = _fitz.Rect(c.x0 - pad, c.y0 - pad, c.x1 + pad, c.y1 + pad)
            merged = False
            for i, nc in enumerate(new_clusters):
                if nc.intersects(expanded):
                    new_clusters[i] = nc | c
                    merged = True
                    changed = True
                    break
            if not merged:
                new_clusters.append(c)
        clusters = new_clusters

    # Discard clusters that cover too much of the page
    clusters = [
        c for c in clusters
        if (c.width * c.height) / page_area <= _MAX_PAGE_COVERAGE
    ]

    return clusters


def extract_figures(
    markdown: str,
    page_image: Image.Image,
    page_num: int,
    pdf_page=None,
    render_dpi: int = 300,
) -> tuple[str, list[tuple[str, Image.Image]]]:
    """Extract figure crops based on coordinate references from the model.

    When a ``pdf_page`` is provided, detects actual figure regions from the
    PDF's graphical elements (images + drawings), matches them to the model's
    references by spatial overlap, and renders from the vector PDF at
    ``render_dpi``.  Falls back to raster cropping for plain images.

    Returns updated markdown and list of (filename, cropped_image) pairs.
    """
    import fitz as _fitz

    figures: list[tuple[str, Image.Image]] = []
    inf_w, inf_h = _inference_size(page_image.size)

    # Pre-detect figure regions from PDF if available
    detected_regions: list = []
    if pdf_page is not None:
        detected_regions = _detect_figure_regions(pdf_page)

    def _replace(match: re.Match) -> str:
        alt = match.group(1)
        mx, my, mw, mh = (
            int(match.group(2)),
            int(match.group(3)),
            int(match.group(4)),
            int(match.group(5)),
        )

        if pdf_page is not None and detected_regions:
            # Vector PDF with detected graphical elements — render from PDF
            page_rect = pdf_page.rect
            sx = page_rect.width / inf_w
            sy = page_rect.height / inf_h
            model_rect = _fitz.Rect(
                mx * sx, my * sy, (mx + mw) * sx, (my + mh) * sy
            )

            # Union all detected regions that overlap the model's estimate
            clip = model_rect
            for region in detected_regions:
                intersection = model_rect & region
                if not intersection.is_empty:
                    clip = clip | region  # union

            # Add padding and clamp to page
            clip = _fitz.Rect(
                clip.x0 - _FIGURE_PADDING_PTS,
                clip.y0 - _FIGURE_PADDING_PTS,
                clip.x1 + _FIGURE_PADDING_PTS,
                clip.y1 + _FIGURE_PADDING_PTS,
            )
            clip = clip & page_rect

            zoom = render_dpi / 72
            mat = _fitz.Matrix(zoom, zoom)
            pix = pdf_page.get_pixmap(matrix=mat, clip=clip)
            cropped = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
        else:
            # Scanned PDF or plain image — crop from raster with adaptive padding
            orig_w, orig_h = page_image.size
            sx = orig_w / inf_w
            sy = orig_h / inf_h
            # Use percentage-based expansion for model's approximate coordinates
            pad_x = int(mw * sx * _RASTER_PAD_FRAC)
            pad_y = int(mh * sy * _RASTER_PAD_FRAC)
            crop_box = (
                max(0, int(mx * sx) - pad_x),
                max(0, int(my * sy) - pad_y),
                min(orig_w, int((mx + mw) * sx) + pad_x),
                min(orig_h, int((my + mh) * sy) + pad_y),
            )
            cropped = page_image.crop(crop_box)

        filename = f"page{page_num}_fig{len(figures) + 1}.png"
        figures.append((filename, cropped))
        return f"![{alt}]({filename})"

    updated = _FIGURE_RE.sub(_replace, markdown)
    return updated, figures


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
    ) -> None:
        self._model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        self._processor = AutoProcessor.from_pretrained(
            model_name,
            min_pixels=28 * 28,
            max_pixels=1280 * 28 * 28,
        )

    def recognize_image(self, image: Image.Image) -> str:
        """Recognize text and formulas in an image. Returns Markdown with LaTeX."""
        image = _resize_for_inference(image)

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": OLMOCR_PROMPT},
                ],
            }
        ]

        text_input = self._processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self._processor(
            text=[text_input],
            images=[image],
            return_tensors="pt",
        ).to(self._model.device)

        with torch.no_grad():
            output_ids = self._model.generate(
                **inputs,
                max_new_tokens=8000,
                temperature=0.1,
                do_sample=True,
            )

        # Trim the input tokens from the output
        generated_ids = output_ids[0, inputs["input_ids"].shape[1]:]
        raw_text = self._processor.decode(
            generated_ids, skip_special_tokens=True
        )
        text = _strip_yaml_front_matter(raw_text).strip()
        return _normalize_latex_delimiters(text)
