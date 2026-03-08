# src/local_ocr/engine.py
import re

from PIL import Image
from vllm import LLM, SamplingParams

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

_CHAT_PROMPT = (
    "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
    "<|im_start|>user\n"
    "<|vision_start|><|image_pad|><|vision_end|>"
    f"{OLMOCR_PROMPT}<|im_end|>\n"
    "<|im_start|>assistant\n"
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
        max_model_len: int = 16384,
    ) -> None:
        self._llm = LLM(
            model=model_name,
            max_model_len=max_model_len,
            max_num_seqs=5,
            mm_processor_kwargs={
                "min_pixels": 28 * 28,
                "max_pixels": 1280 * 28 * 28,
            },
        )
        self._sampling_params = SamplingParams(
            temperature=0.1,
            max_tokens=8000,
        )

    def recognize_image(self, image: Image.Image) -> str:
        """Recognize text and formulas in an image. Returns Markdown with LaTeX."""
        image = _resize_for_inference(image)
        outputs = self._llm.generate(
            [{"prompt": _CHAT_PROMPT, "multi_modal_data": {"image": image}}],
            sampling_params=self._sampling_params,
        )
        raw_text = outputs[0].outputs[0].text
        return _strip_yaml_front_matter(raw_text).strip()
