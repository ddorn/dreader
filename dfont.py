from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import pygame

from style import FontStyle


def split_keep(text: str) -> list[str]:
    """Split a text into words and spaces, so that "".join(parts) == text"""
    # Either full spaces or full non-spaces.
    return re.findall(r"\s+|\S+", text)


@dataclass
class TextParts:
    parts: list[tuple[str, pygame.Rect]]

    def __post_init__(self):
        assert self.parts

    @property
    def width(self):
        return max(part[1].right for part in self.parts)

    @property
    def height(self):
        return sum(part[1].height for part in self.parts)

    @property
    def bottom(self):
        return self.parts[-1][1].bottom

    @property
    def continuation_pos(self):
        return self.parts[-1][1].topright

    def shift(self, dx: float, dy: float):
        return self.__class__([(text, rect.move(dx, dy)) for text, rect in self.parts])


def wrap(
    text: str, font: pygame.Font, max_width: int | None = None, leading_space: int = 0
) -> TextParts:
    """Return the size of the (optionally wrapped) text, including with newlines."""

    lines = text.splitlines()
    line_height = font.get_height()

    if not lines:
        return TextParts([("", pygame.Rect(leading_space, 0, 0, line_height))])

    if max_width is None:
        sizes = [font.size(line) for line in lines]
        return TextParts(
            [
                (
                    line,
                    pygame.Rect(
                        0 if i > 0 else leading_space, i * line_height, size[0], line_height
                    ),
                )
                for i, (line, size) in enumerate(zip(lines, sizes))
            ]
        )

    if lines == [""]:
        return TextParts([("", pygame.Rect(leading_space, 0, 0, line_height))])

    y = 0
    skipped_lines = 0
    parts = []
    for input_line in lines:
        blocks = split_keep(input_line)

        if not blocks:
            leading_space = 0
            skipped_lines += 1
            continue
        skipped_lines = 0

        while blocks:
            visual_line = []
            # Try adding blocks until the line is too long.
            for block in blocks:
                if font.size("".join(visual_line + [block]))[0] > max_width - leading_space:
                    break
                visual_line.append(block)

            blocks = blocks[len(visual_line) :]

            # If next block is a space, ignore it.
            if blocks and blocks[0].isspace():
                blocks.pop(0)

            # If we have a block that is too long, we need to split it.
            if blocks and not visual_line:
                to_split = blocks.pop(0)
                # print(f"Splitting {to_split!r}")
                # Try to split on letters/vs non-letters.
                sub_words = re.findall(r"\w+|\W+", to_split)
                if len(sub_words) > 1:
                    blocks = sub_words + blocks
                    continue
                else:
                    # Find the longest split that fits.
                    if to_split.isalnum():
                        suffix = "-"
                    else:
                        suffix = ""

                    for i in range(len(to_split)):
                        if font.size(to_split[:i] + suffix)[0] > max_width - leading_space:
                            break

                    if i > 0:
                        visual_line.append(to_split[: i - 1] + suffix)
                        blocks.insert(0, to_split[i - 1 :])
                    else:
                        # Abort, we can't split this block. Maybe too much indend.
                        # Give up and overflow.
                        # print(f"Failed to split {to_split!r}")
                        visual_line.append(to_split)

            visual_line = "".join(visual_line)
            parts.append(
                (
                    visual_line,
                    pygame.Rect(leading_space, y, *font.size(visual_line)),
                )
            )
            y += line_height
            # We only need to add leading space after the first line.
            leading_space = 0

    if skipped_lines:
        parts.append(("", pygame.Rect(leading_space, y, 0, line_height)))

    return TextParts(parts)


@staticmethod
@lru_cache(300)
def _get_font(path: Path | str | None, size: int, bold: bool, italic: bool) -> pygame.Font:
    if path is None:
        pass
    elif not Path(path).exists():
        # We assume it's a font name
        path = pygame.font.match_font(path)  # , bold=bold, italic=italic)
        print(f"Matched to {path}")
    font = pygame.Font(path, size)
    font.bold = bold
    font.italic = italic
    return font


def get_font(
    path: Path | str | None, size: int | float = 30, bold: bool = False, italic: bool = False
) -> pygame.Font:
    # We have this to provide defaults without having multiple entries in the cache,
    # plus to ensure the size is an int (for the same reason (and pygame probably preferse it too)).
    return _get_font(path, int(size), bold, italic)


def list_to_tuple(obj):
    if isinstance(obj, list):
        return tuple(obj)
    return obj


def render(text: str, style: FontStyle):
    size = int(style.size)
    color = list_to_tuple(style.color)
    background = list_to_tuple(style.background)
    underline = list_to_tuple(style.underline)

    return _render(
        text,
        size,
        color,
        style.file,
        background=background,
        underline=underline,
        bold=style.bold,
        italic=style.italic,
    )


def render_simple(text, size, color=(0, 0, 0), font_path=None):
    return _render(
        text, size, color, font_path, background=None, underline=None, bold=False, italic=False
    )


@lru_cache(1000)
def _render(
    text: str,
    size: int,
    color: tuple,
    font_path: Path,
    *,
    background: tuple | None,
    underline: tuple[int, int, int] | None,
    bold: bool,
    italic: bool,
):

    font = get_font(font_path, size, bold=bold, italic=italic)

    sizing = wrap(text, font=font)

    surf = pygame.Surface((sizing.width, sizing.height), pygame.SRCALPHA)
    for part, rect in sizing.parts:
        s = font.render(part, True, color, background)
        if underline:
            baseline = font.get_ascent() + 2
            pygame.draw.line(s, underline, (0, baseline), (s.get_width(), baseline), 1)
        surf.blit(s, rect)

    return surf
