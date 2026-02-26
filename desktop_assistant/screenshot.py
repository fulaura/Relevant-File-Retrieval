from __future__ import annotations

from pathlib import Path

import mss
from PIL import Image


def capture_fullscreen(output_path: Path) -> Path:
    """Capture the primary monitor and save to the output path."""
    with mss.mss() as sct:
        shot = sct.shot(mon=-1, output=str(output_path))
    return Path(shot)


def load_image(path: Path) -> Image.Image:
    return Image.open(path)
