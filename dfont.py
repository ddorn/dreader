from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from pprint import pprint
import re
import pygame
import pygame.locals as pg


def blit_aligned(
    surf: pygame.Surface,
    to_blit: pygame.Surface,
    y: int,
    align: int = pg.FONT_LEFT,
    left: int | None = None,
    right: int | None = None,
) -> pygame.Rect:
    if left is None:
        left = 0
    if right is None:
        right = surf.get_width()

    if align == pg.FONT_LEFT:
        return surf.blit(to_blit, (left, y))
    elif align == pg.FONT_RIGHT:
        return surf.blit(to_blit, (right - to_blit.get_width(), y))
    elif align == pg.FONT_CENTER:
        return surf.blit(to_blit, ((left + right - to_blit.get_width()) // 2, y))
    else:
        raise ValueError(f"Invalid alignment: {align}")


def split_keep_trailing_spaces(text: str):
    """Split a text into parts, so that " ".join(parts) == text if there are no consecutive spaces."""
    # Either full spaces or full non-spaces.
    raw_parts = re.split(r"(\s+|\S+)", text)
    parts = [part for part in raw_parts if part.strip()]
    # Re-add spaces at the extremities.
    if len(raw_parts) >= 2:
        if not raw_parts[0].strip():
            parts.insert(0, " ")
        if not raw_parts[-1].strip():
            parts.append(" ")
    elif len(raw_parts) == 1 and not raw_parts[0].strip():
        parts.append(" ")
    return parts


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
    def continuation_pos(self):
        return self.parts[-1][1].topright

    def shift(self, dx: int, dy: int):
        return self.__class__([(text, rect.move(dx, dy)) for text, rect in self.parts])


class DFont:
    def __init__(self, path: Path):
        self.path = path
        self.by_size: dict[tuple, pygame.Font] = {}

    def get_font(self, size: int | float, align: int = pg.FONT_LEFT) -> pygame.Font:
        size = int(size)
        key = size, align
        if key not in self.by_size:
            self.by_size[key] = pygame.Font(self.path, size)
            self.by_size[key].align = align
        return self.by_size[key]

    __call__ = get_font

    @lru_cache(1000)
    def render(
        self,
        text: str,
        size: int | tuple[int, int],
        color: tuple,
        align: int = pg.FONT_LEFT,
    ):
        if not isinstance(size, int):
            size = self.auto_size(text, size)

        font = self.get_font(size, align)

        sizing = self.size(text, size)

        surf = pygame.Surface((sizing.width, sizing.height), pygame.SRCALPHA)
        for part, rect in sizing.parts:
            s = font.render(part, True, color)
            blit_aligned(surf, s, rect.y, align)

        return surf

    def size(
        self, text: str, size: int, max_width: int | None = None, leading_space: int = 0
    ) -> TextParts:
        """Return the size of the (optionally wrapped) text, including with newlines."""

        lines = text.splitlines()
        font = self.get_font(size)
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

        parts = []
        for line in lines:

            if line.strip() == "":
                parts.append(
                    ("", pygame.Rect(leading_space, len(parts) * line_height, 0, line_height))
                )
                leading_space = 0
                continue

            words = line.split()
            while words:
                visual_line = []
                while font.size(" ".join(visual_line))[0] < max_width - leading_space:
                    if not words:
                        break
                    visual_line.append(words.pop(0))
                else:
                    words.insert(0, visual_line.pop())
                visual_line = " ".join(visual_line)
                parts.append(
                    (
                        visual_line,
                        pygame.Rect(
                            leading_space, len(parts) * line_height, *font.size(visual_line)
                        ),
                    )
                )
                # We only need to add leading space after the first line.
                leading_space = 0

        return TextParts(parts)

    def auto_size(self, text: str, max_rect: tuple[int, int]):
        """Find the largest font size that will fit text in max_rect."""
        # Use dichotomy to find the largest font size that will fit text in max_rect.

        min_size = 1
        max_size = max_rect[1]
        while min_size < max_size:
            font_size = (min_size + max_size) // 2
            text_size = self.size(text, font_size)

            if text_size.width <= max_rect[0] and text_size.height <= max_rect[1]:
                min_size = font_size + 1
            else:
                max_size = font_size
        return min_size - 1

    def table(
        self,
        rows: list[list[str]],
        size: int | tuple[int, int],
        color: tuple[int, int, int] | list[tuple[int, int, int]],
        title: str | None = None,
        col_sep: str = "__",
        align: int | list[int] = pg.FONT_LEFT,
        title_color: tuple[int, int, int] | None = None,
        title_align: int = pg.FONT_CENTER,
        hidden_rows: list[list[str]] = [],
        header_line_color: tuple[int, int, int] | None = None,
    ):
        """Render a table with the given rows and size.

        Args:
            rows: The rows of the table.
            size: The font size of the table. If this is a tuple, the table is the largest that can fit in this (width, height).
            color: The text color of each column. If this is a tuple, it is used for all columns.
            title: The optional title of the table.
            col_sep: Text whose width will be used to separate columns.
            align: The alignment of each column. If this is an int, it is be used for all columns.
            title_color: The color of the title. If omitted, the color of the first column is be used.
            hidden_rows: Rows that are not rendered, but are used to size the table. Prevents change of size when scrolling.
            header_line_color: Draw a line after the first row with this color.
        """
        assert rows

        cols = list(zip(*rows, strict=True))

        if isinstance(align, int):
            align = [align] * len(cols)
        if isinstance(color, tuple):
            color = [color] * len(cols)
        assert len(align) == len(cols)
        assert len(color) == len(cols)
        if title_color is None:
            title_color = color[0]

        # It's a bit hard to size a table, we do it by creating a dummy text
        # block that has the same size.
        dummy_font = self.get_font(10)  # len() is not a good proxy for visual size.
        cols_with_hidden = list(zip(*rows, *hidden_rows, strict=True))
        longest_by_col = [max(col, key=lambda x: dummy_font.size(x)[0]) for col in cols_with_hidden]
        long_line = col_sep.join(longest_by_col)
        dummy_long_content = "\n".join([long_line] * len(rows))
        if title:
            dummy_long_content = title + "\n" + dummy_long_content

        if not isinstance(size, int):
            size = self.auto_size(dummy_long_content, size)

        font = self.get_font(size)
        surf = font.render(dummy_long_content, True, (0, 0, 0))
        surf.fill((0, 0, 0, 0))

        # Draw title
        if title:
            title_surf = font.render(title, True, title_color)
            y = blit_aligned(surf, title_surf, 0, title_align).bottom
        else:
            y = 0

        sep_width = font.size(col_sep)[0]
        column_widths = [font.size(longest)[0] for longest in longest_by_col]

        # Render each column
        x = 0
        for col, align, col_color, width in zip(cols, align, color, column_widths):
            col_surf = self.get_font(size, align).render("\n".join(col), True, col_color)
            blit_aligned(surf, col_surf, y, align, x, x + width)
            x += width + sep_width

        # Draw a line under the header
        if header_line_color is not None:
            y += font.get_height()
            pygame.draw.line(surf, header_line_color, (0, y), (surf.get_width(), y), 1)

        return surf
