from __future__ import annotations

from dataclasses import dataclass
import dataclasses
from pathlib import Path
from pprint import pprint
import re
from typing import Generator

import pygame
import marko.block
import marko.inline
import yaml
import bisect


from dfont import DFont, TextParts


CSS = yaml.safe_load(Path("style.yaml").read_text())
SIZE_ATTRIBUTES = {"font_size"}
RE_SIZE = re.compile(r"([\d.]+)(\w+)")


@dataclass
class Style:
    font_obj: DFont
    base_size: int
    color: tuple[int, int, int]
    font_size: int = 1

    classes: set[str] = dataclasses.field(default_factory=set)

    def with_class(self, cls: str) -> Style:
        attrs = dict(self.__dict__)
        attrs["classes"] = self.classes | {cls}
        return Style(**attrs)

    def without_class(self, cls: str) -> Style:
        attrs = dict(self.__dict__)
        attrs["classes"] = self.classes - {cls}
        return Style(**attrs)

    __add__ = with_class
    __sub__ = without_class

    def __getattribute__(self, name: str):
        base = super().__getattribute__(name)
        classes = super().__getattribute__("classes")

        for cls in classes:
            base = CSS[cls].get(name, base)

        if name in SIZE_ATTRIBUTES:
            if isinstance(base, str):
                match = RE_SIZE.match(base)
                if match:
                    size, unit = match.groups()
                    base = float(size) * super().__getattribute__("base_size")
                    if unit == "%":
                        base = base // 100
                    elif unit == "rem":
                        pass
                    else:
                        raise ValueError(f"Unknown unit: {unit}")
                else:
                    raise ValueError(f"Invalid size: {base}")
            elif isinstance(base, (int, float)):
                base = base * super().__getattribute__("base_size")
            else:
                raise TypeError(f"Invalid type for {name}: {type(base)}")
            base = int(base)
        return base


class InlineText:
    def __init__(
        self,
        text: str,
        style: Style,
        hard_break: bool = False,
    ):
        self.text = text
        self.style = style
        self.hard_break = hard_break

        # Set by layout()
        self.size = (0, 0)
        self.text_parts: TextParts = None  # type: ignore
        self.continuation_pos = (0, 0)

    def __repr__(self):
        short_text = self.text[:20] + "..." if len(self.text) > 23 else self.text
        show = dict(
            text=repr(short_text),
            size=self.size,
            parts=self.text_parts,
            cont=self.continuation_pos,
            hard_break=self.hard_break,
        )
        as_str = ", ".join(f"{k}={v}" for k, v in show.items())
        return f"<InlineText {as_str}>"

    def layout(self, width: float, type_head: tuple[int, int]):
        metrics = self.style.font_obj.size(
            self.text, self.style.font_size, int(width), type_head[0]
        )
        metrics = metrics.shift(0, type_head[1])
        self.text_parts = metrics
        self.size = metrics.width, metrics.height
        if self.hard_break:
            self.continuation_pos = (0, metrics.bottom)
        else:
            self.continuation_pos = metrics.continuation_pos
        self.rect = pygame.Rect(0, type_head[1], *self.size)

    def render(self, scroll_x: int, scroll_y: int, screen):
        rect_on_screen = self.rect.move(scroll_x, scroll_y)
        if not rect_on_screen.colliderect(screen.get_rect()):
            return

        debug = pygame.key.get_pressed()[pygame.K_BACKSLASH]

        for text, rect in self.text_parts.parts:
            surf = self.style.font_obj.render(text, self.style.font_size, self.style.color)
            screen.blit(surf, (rect.x + scroll_x, rect.y + scroll_y))

            if debug:
                r = pygame.Rect(rect.x + scroll_x, rect.y + scroll_y, rect.width, rect.height)
                pygame.draw.rect(screen, (255, 0, 0), r, 1)
                pygame.draw.rect(screen, (0, 255, 0), r.inflate(5, 5), 1)
                s = self.style.font_obj(20).render(repr(text), True, (255, 0, 0))
                screen.blit(s, (rect.x + scroll_x, rect.y + scroll_y))

    def is_before(self, x, y):
        """Check is a given point is after the text in the document."""

        # If it's further right & down of the topleft of the first part, it's after
        first_part = self.text_parts.parts[0][1]
        if first_part.x < x and first_part.y < y:
            return True

        # If it's lower than the bottom of the first part, it's after
        if first_part.bottom < y:
            return True

        return False


class Document:

    def __init__(self, children: list[InlineText]) -> None:
        self.size = (0, 0)
        self.children = children

    def layout(self, width: float):
        write_head = (0, 0)
        for child in self.children:
            child.layout(width, write_head)
            write_head = child.continuation_pos

        self.size = max(child.size[0] for child in self.children), sum(
            child.size[1] for child in self.children
        )

    def render(self, x, y, screen):
        for child in self.children:
            child.render(x, y, screen)

    @classmethod
    def from_marko(cls, doc, style: Style):
        return cls(list(from_marko(doc, style)))

    def at(self, x, y) -> tuple[int, InlineText]:
        # Find the first child that is after the point
        i = 0
        while i < len(self.children) and self.children[i].is_before(x, y):
            i += 1

        if i > 0:
            i -= 1
        return i, self.children[i]


def flatten(node, style: Style) -> Generator[InlineText, None, None]:
    """Flatten a document tree into a list of InlineText objects."""

    if isinstance(node, marko.block.Document):
        for child in node.children:
            yield from flatten(child, style)
    elif isinstance(node, marko.block.Paragraph):
        for child in node.children:
            yield from flatten(child, style)
    elif isinstance(node, marko.block.BlankLine):
        yield InlineText("\n", style, hard_break=True)
    elif isinstance(node, marko.block.Heading):
        style = style.with_class(f"h{node.level}")
        for child in node.children:
            yield from flatten(child, style)
        yield InlineText("", style, hard_break=True)
    elif isinstance(node, marko.block.List):
        for i, child in enumerate(node.children, start=node.start):
            if node.ordered:
                prefix = f"{i}. "
            else:
                prefix = node.bullet + " "
            yield InlineText(prefix, style)
            yield from flatten(child, style)
    elif isinstance(node, marko.block.ListItem):
        for child in node.children:
            yield from flatten(child, style)
        yield InlineText("", style, hard_break=True)
    elif isinstance(node, marko.inline.RawText):
        yield InlineText(node.children, style)
    elif isinstance(node, marko.inline.LineBreak):
        if node.soft:
            yield InlineText(" ", style)
        else:
            yield InlineText("", style, hard_break=True)
    else:
        yield InlineText(
            f"Unsupported element: {node.__class__.__name__}", style.with_class("error")
        )


def from_marko(doc, style: Style) -> Generator[InlineText, None, None]:
    # The main goal here is to filter redundant newlines.
    # We want to keep only one newline in a row.

    last_was_hard_break = False
    for child in flatten(doc, style):
        if last_was_hard_break and child.hard_break and not child.text.strip():
            continue
        last_was_hard_break = child.hard_break
        yield child
