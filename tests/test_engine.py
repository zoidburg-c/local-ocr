# tests/test_engine.py
from unittest.mock import MagicMock, patch
import torch
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
