# tests/test_engine.py
from unittest.mock import MagicMock, patch
import torch
from PIL import Image

from local_ocr.engine import (
    OCREngine,
    OLMOCR_PROMPT,
    _strip_yaml_front_matter,
    _normalize_latex_delimiters,
    _wrap_multiline_display_math,
    extract_figures,
)


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


@patch("local_ocr.engine.AutoProcessor")
@patch("local_ocr.engine.Qwen2_5_VLForConditionalGeneration")
def test_recognize_image(mock_model_cls, mock_proc_cls):
    mock_processor = MagicMock()
    mock_processor.apply_chat_template.return_value = "formatted prompt"
    mock_inputs = MagicMock()
    mock_inputs.__getitem__ = MagicMock(
        side_effect=lambda k: torch.tensor([[1, 2, 3]]) if k == "input_ids" else None
    )
    mock_inputs.to.return_value = mock_inputs
    mock_processor.return_value = mock_inputs
    mock_processor.decode.return_value = "# Question 1\n\nSolve $x^2 = 4$"
    mock_proc_cls.from_pretrained.return_value = mock_processor

    mock_model = MagicMock()
    mock_model.device = "cpu"
    mock_model.generate.return_value = torch.tensor([[1, 2, 3, 4, 5, 6]])
    mock_model_cls.from_pretrained.return_value = mock_model

    engine = OCREngine()
    image = Image.new("RGB", (100, 100), "white")
    result = engine.recognize_image(image)

    assert "# Question 1" in result
    assert "$x^2 = 4$" in result
    mock_model.generate.assert_called_once()


@patch("local_ocr.engine.AutoProcessor")
@patch("local_ocr.engine.Qwen2_5_VLForConditionalGeneration")
def test_recognize_image_strips_front_matter(mock_model_cls, mock_proc_cls):
    raw_output = (
        '---\nprimary_language: "en"\nis_table: false\n---\n'
        "# Math Exam\n\n$$\\int_0^1 x^2 dx$$"
    )
    mock_processor = MagicMock()
    mock_processor.apply_chat_template.return_value = "prompt"
    mock_inputs = MagicMock()
    mock_inputs.__getitem__ = MagicMock(
        side_effect=lambda k: torch.tensor([[1, 2, 3]]) if k == "input_ids" else None
    )
    mock_inputs.to.return_value = mock_inputs
    mock_processor.return_value = mock_inputs
    mock_processor.decode.return_value = raw_output
    mock_proc_cls.from_pretrained.return_value = mock_processor

    mock_model = MagicMock()
    mock_model.device = "cpu"
    mock_model.generate.return_value = torch.tensor([[1, 2, 3, 4, 5, 6]])
    mock_model_cls.from_pretrained.return_value = mock_model

    engine = OCREngine()
    image = Image.new("RGB", (100, 100), "white")
    result = engine.recognize_image(image)

    assert result.startswith("# Math Exam")
    assert "---" not in result


@patch("local_ocr.engine.AutoProcessor")
@patch("local_ocr.engine.Qwen2_5_VLForConditionalGeneration")
def test_recognize_image_resizes_large_image(mock_model_cls, mock_proc_cls):
    mock_processor = MagicMock()
    mock_processor.apply_chat_template.return_value = "prompt"
    mock_inputs = MagicMock()
    mock_inputs.__getitem__ = MagicMock(
        side_effect=lambda k: torch.tensor([[1, 2, 3]]) if k == "input_ids" else None
    )
    mock_inputs.to.return_value = mock_inputs
    mock_processor.return_value = mock_inputs
    mock_processor.decode.return_value = "content"
    mock_proc_cls.from_pretrained.return_value = mock_processor

    mock_model = MagicMock()
    mock_model.device = "cpu"
    mock_model.generate.return_value = torch.tensor([[1, 2, 3, 4, 5, 6]])
    mock_model_cls.from_pretrained.return_value = mock_model

    engine = OCREngine()
    large_image = Image.new("RGB", (4000, 3000), "white")
    engine.recognize_image(large_image)

    # Verify the images passed to processor were resized
    call_args = mock_processor.call_args
    passed_images = call_args[1]["images"]
    assert max(passed_images[0].size) <= 1288


@patch("local_ocr.engine.AutoProcessor")
@patch("local_ocr.engine.Qwen2_5_VLForConditionalGeneration")
def test_prompt_contains_latex_instruction(mock_model_cls, mock_proc_cls):
    mock_processor = MagicMock()
    mock_processor.apply_chat_template.return_value = "prompt"
    mock_inputs = MagicMock()
    mock_inputs.__getitem__ = MagicMock(
        side_effect=lambda k: torch.tensor([[1, 2, 3]]) if k == "input_ids" else None
    )
    mock_inputs.to.return_value = mock_inputs
    mock_processor.return_value = mock_inputs
    mock_processor.decode.return_value = "content"
    mock_proc_cls.from_pretrained.return_value = mock_processor

    mock_model = MagicMock()
    mock_model.device = "cpu"
    mock_model.generate.return_value = torch.tensor([[1, 2, 3, 4, 5, 6]])
    mock_model_cls.from_pretrained.return_value = mock_model

    engine = OCREngine()
    image = Image.new("RGB", (100, 100), "white")
    engine.recognize_image(image)

    # Verify the prompt passed to apply_chat_template includes LaTeX instruction
    call_args = mock_processor.apply_chat_template.call_args
    messages = call_args[0][0]
    text_content = [c for c in messages[0]["content"] if c["type"] == "text"][0]["text"]
    assert "LateX" in text_content or "LaTeX" in text_content


def test_normalize_latex_delimiters():
    text = r"Solve \( x^2 = 4 \) and \[ \int_0^1 x\,dx \]"
    result = _normalize_latex_delimiters(text)
    assert result == r"Solve $x^2 = 4$ and $$ \int_0^1 x\,dx $$"


def test_normalize_latex_delimiters_no_change():
    text = "Solve $x^2 = 4$ and $$\\int_0^1 x\\,dx$$"
    result = _normalize_latex_delimiters(text)
    assert result == text


def test_normalize_latex_strips_spaces_from_native_dollar():
    text = "Solve $ x^2 = 4 $ and $$ \\int_0^1 x\\,dx $$"
    result = _normalize_latex_delimiters(text)
    # Inline spaces stripped, display math spaces preserved
    assert result == "Solve $x^2 = 4$ and $$ \\int_0^1 x\\,dx $$"


def test_wrap_multiline_display_math():
    text = "$$\n|z| + 5w = 0 \\\\\niz - 4w = -4 + 7i\n$$"
    result = _wrap_multiline_display_math(text)
    assert r"\begin{aligned}" in result
    assert r"\end{aligned}" in result
    assert "|z| + 5w = 0 \\\\" in result


def test_wrap_multiline_display_math_skips_existing_env():
    text = "$$\n\\begin{cases}\na, & x > 0 \\\\\nb, & x \\leq 0\n\\end{cases}\n$$"
    result = _wrap_multiline_display_math(text)
    assert result == text


def test_wrap_multiline_display_math_skips_single_line():
    text = "$$\nx^2 + y^2 = 1\n$$"
    result = _wrap_multiline_display_math(text)
    assert result == text


def test_extract_figures_raster_fallback():
    """When no pdf_page, crops from raster with coordinate scaling."""
    markdown = "Some text\n![A graph](page_100_200_300_150.png)\nMore text"
    image = Image.new("RGB", (800, 600), "white")
    updated, figures = extract_figures(markdown, image, page_num=3)
    assert "![A graph](page3_fig1.png)" in updated
    assert len(figures) == 1
    assert figures[0][0] == "page3_fig1.png"
    assert figures[0][1].size == (300, 150)


def test_extract_figures_raster_scales_coordinates():
    """Coordinates from resized inference image are scaled to original size."""
    markdown = "![Fig](page_100_200_300_150.png)"
    image = Image.new("RGB", (2550, 3300), "white")
    updated, figures = extract_figures(markdown, image, page_num=1)
    assert len(figures) == 1
    # inference size: 995x1288; scale factors: 2550/995, 3300/1288
    crop = figures[0][1]
    assert crop.size[0] > 300  # scaled up from inference coords
    assert crop.size[1] > 150


def test_extract_figures_no_figures():
    markdown = "No figures here"
    image = Image.new("RGB", (800, 600), "white")
    updated, figures = extract_figures(markdown, image, page_num=1)
    assert updated == markdown
    assert figures == []
