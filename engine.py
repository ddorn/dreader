from __future__ import annotations

from abc import abstractmethod, ABC
from dataclasses import dataclass
from typing import Generator, Self
import warnings

import pygame
import marko.block
import marko.inline

from dfont import DFont, TextParts


@dataclass
class Style:
    font_obj: DFont
    font_size: int
    color: tuple[int, int, int]
    header_multipliers: tuple[float, float, float, float, float, float] = (
        2.0,
        1.5,
        1.17,
        1.12,
        1.0,
        1.0,
    )

    def clone(self, **kwargs):
        kwargs = {**self.__dict__, **kwargs}
        return Style(**kwargs)


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
                pygame.draw.rect(
                    screen,
                    (255, 0, 0),
                    (rect.x + scroll_x, rect.y + scroll_y, rect.width + 1, rect.height),
                    1,
                )
                s = self.style.font_obj(10).render(repr(text), True, (255, 0, 0))
                screen.blit(s, (rect.x + scroll_x, rect.y + scroll_y))


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
        style = style.clone(
            font_size=int(style.font_size * style.header_multipliers[node.level - 1])
        )
        for child in node.children:
            yield from flatten(child, style)
        yield InlineText("", style, hard_break=True)
    elif isinstance(node, marko.inline.RawText):
        yield InlineText(node.children, style)
    elif isinstance(node, marko.inline.LineBreak):
        if node.soft:
            yield InlineText(" ", style.clone(color=(0, 0, 250)))
        else:
            yield InlineText("", style, hard_break=True)
    else:
        yield InlineText(
            f"Unsupported element: {node.__class__.__name__}", style.clone(color=(255, 0, 0))
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
