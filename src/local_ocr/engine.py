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

_FRONT_MATTER_RE = re.compile(r"^---\s*\n.*?\n---\s*\n", re.DOTALL)


def _strip_yaml_front_matter(text: str) -> str:
    """Remove YAML front matter block from OlmOCR-2 output."""
    return _FRONT_MATTER_RE.sub("", text)


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
        return _strip_yaml_front_matter(raw_text).strip()
