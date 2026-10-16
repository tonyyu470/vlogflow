from __future__ import annotations

import math
import re
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

FONT = Path(__file__).parent / "assets" / "NotoSansSC.ttf"


def wrap(text: str, font, width: int) -> list[str]:
    words = re.findall(r"[A-Za-z0-9]+(?:[’\'-][A-Za-z0-9]+)*|[^\x00-\x7f]|[^\w\s]|\s+", text)
    lines = []
    line = ""
    for word in words:
        if "\n" in word:
            lines.append(line.strip())
            line = ""
            continue
        # Split an unusually long URL/token as well as ordinary CJK text.
        pieces = list(word) if font.getlength(word) > width else [word]
        for part in pieces:
            candidate = line + part
            if font.getlength(candidate) > width and line:
                lines.append(line.strip())
                line = part.lstrip()
            else:
                line = candidate
    if line.strip():
        lines.append(line.strip())
    return lines


def layout_caption(zh: str, en: str, width: int, height: int):
    texts = [t.strip() for t in (zh, en) if t.strip()]
    if not texts:
        raise ValueError("Caption text is empty")
    size = max(24, round(width * 0.046))
    while size >= 16:
        font = ImageFont.truetype(str(FONT), size)
        lines = [line for t in texts for line in wrap(t, font, width - int(width * 0.14))]
        line_height = round(size * 1.42)
        if len(lines) * line_height <= height * 0.32:
            break
        size -= 2
    if size < 16:
        raise ValueError("Caption is too long; split or shorten the scene text")
    total = len(lines) * line_height
    bottom = height - round(height * 0.09)
    top = bottom - total
    left = round(width * 0.05)
    right = width - left
    img = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    draw.rounded_rectangle((left, top - 10, right, bottom + 10), radius=12, fill=(0, 0, 0, 165))
    for n, line in enumerate(lines):
        draw.text(
            (width / 2, top + n * line_height),
            line,
            font=font,
            fill="white",
            anchor="mt",
            stroke_width=1,
            stroke_fill=(0, 0, 0, 210),
        )
    return img, (left, top - 10, right, bottom + 10)


def selected(cue, mode):
    return (
        "\n".join(cue.get(k, "").strip() for k in ("zh", "en") if cue.get(k, "").strip())
        if mode == "bilingual"
        else cue.get(mode, "").strip()
    )


def validate_cues(cues, duration):
    previous = 0
    for c in cues:
        if (
            not all(math.isfinite(c[k]) for k in ("start", "end"))
            or c["start"] < previous - 0.002
            or c["end"] <= c["start"]
            or c["end"] > duration + 0.002
        ):
            raise ValueError("Subtitle timing overlaps or exceeds the final video")
        previous = c["end"]


def timestamp(value):
    ms = round(value * 1000)
    h, ms = divmod(ms, 3600000)
    m, ms = divmod(ms, 60000)
    s, ms = divmod(ms, 1000)
    return f"{h:02}:{m:02}:{s:02},{ms:03}"


def write_srt(cues, mode, path: Path, duration: float):
    validate_cues(cues, duration)
    if any(not selected(c, mode) for c in cues):
        raise ValueError("Selected subtitle language has missing translations")
    path.write_text(
        "\n\n".join(
            f"{i + 1}\n{timestamp(c['start'])} --> {timestamp(c['end'])}\n{selected(c, mode)}"
            for i, c in enumerate(cues)
        )
        + "\n",
        encoding="utf-8",
    )
