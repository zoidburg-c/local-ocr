from pathlib import Path

import numpy as np
from PIL import Image
from pix2text import Pix2Text

from local_ocr.schemas import Element

# Map Pix2Text types to our schema types
_TYPE_MAP = {
    "text": "text",
    "isolated": "formula",
    "embedding": "formula",
}


def _position_to_bbox(position: np.ndarray) -> list[float]:
    """Convert 4-corner points [[x,y],...] to [x_min, y_min, width, height]."""
    x_min = float(position[:, 0].min())
    y_min = float(position[:, 1].min())
    x_max = float(position[:, 0].max())
    y_max = float(position[:, 1].max())
    return [x_min, y_min, x_max - x_min, y_max - y_min]


class OCREngine:
    def __init__(self) -> None:
        self._p2t = Pix2Text()

    def recognize_image(self, image: str | Path | Image.Image) -> list[Element]:
        """Recognize text and formulas in an image. Returns list of Elements."""
        results = self._p2t.recognize_text_formula(image, return_text=False)
        elements = []
        for item in results:
            elem_type = _TYPE_MAP.get(item["type"], "text")
            text = item.get("text", "")
            elements.append(
                Element(
                    type=elem_type,
                    content=text,
                    latex=text if elem_type == "formula" else None,
                    position=_position_to_bbox(item["position"]),
                    score=item.get("score"),
                )
            )
        return elements
