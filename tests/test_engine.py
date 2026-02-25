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
