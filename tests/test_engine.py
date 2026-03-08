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
