from __future__ import annotations

from pathlib import Path
from typing import Optional

import pytesseract
from PIL import Image


class OCRService:
    def __init__(self, tesseract_cmd: Optional[str] = None, language: str = "eng") -> None:
        if tesseract_cmd:
            pytesseract.pytesseract.tesseract_cmd = tesseract_cmd
        self.language = language

    def extract_text(self, image: Image.Image | Path) -> str:
        if isinstance(image, Path):
            image = Image.open(image)
        return pytesseract.image_to_string(image, lang=self.language)
