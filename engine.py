from __future__ import annotations

from dataclasses import dataclass
import dataclasses
from pathlib import Path
from pprint import pprint
import re
from typing import Counter, Generator
import warnings

import marko.ast_renderer
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
    base_size: int = 30
    color: tuple[int, int, int] = (0, 0, 0)
    background: tuple[int, int, int] | None = None
    font_size: int = 1
    underline: tuple[int, int, int] | None = None

    classes: list[str] = dataclasses.field(default_factory=list)

    def with_class(self, cls: str) -> Style:
        attrs = dict(self.__dict__)
        if cls not in self.classes:
            attrs["classes"] = self.classes + [cls]
        else:
            attrs["classes"] = list(self.classes)
        return Style(**attrs)

    def without_class(self, cls: str) -> Style:
        attrs = dict(self.__dict__)
        attrs["classes"] = [c for c in self.classes if c != cls]
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
        link_to: str | None = None,
        anchor: str | None = None,
        indent: int = 0,
    ):
        self.text = text
        self.style = style
        self.hard_break = hard_break
        self.link_to = link_to
        self.anchor = anchor
        self.indent = indent

        # Set by layout()
        self.size = (0, 0)
        self.text_parts: TextParts = None  # type: ignore
        self.continuation_pos = (0, 0)

    def __repr__(self):
        attrs = dict(self.__dict__)
        attrs["text"] = self.text[:20] + "..." if len(self.text) > 23 else self.text
        del attrs["style"]
        attrs["classes"] = self.style.classes

        s = ""
        for key, value in attrs.items():
            if value is True:
                s += f", {key}"
            elif value:
                s += f", {key}={value!r}"

        return f"<InlineText {s[2:]}>"

    def layout(self, indent: float, width: float, type_head: tuple[int, int]):
        metrics = self.style.font_obj.size(
            self.text, self.style.font_size, int(width), type_head[0]
        )
        metrics = metrics.shift(0, type_head[1])
        # Move each part by indent, if it is at the start of the line
        for part in metrics.parts:
            if part[1].x == 0:
                part[1].x += indent
        self.text_parts = metrics
        self.size = metrics.width, metrics.height
        if self.hard_break:
            self.continuation_pos = (0, metrics.bottom)
        else:
            self.continuation_pos = metrics.continuation_pos
        self.rect = pygame.Rect(indent, type_head[1], *self.size)

    def render(self, scroll_x: int, scroll_y: int, screen):
        rect_on_screen = self.rect.move(scroll_x, scroll_y)
        if not rect_on_screen.colliderect(screen.get_rect()):
            return

        debug = pygame.key.get_pressed()[pygame.K_BACKSLASH]

        for text, rect in self.text_parts.parts:
            surf = self.style.font_obj.render(
                text,
                self.style.font_size,
                self.style.color,
                background=self.style.background,
                underline=self.style.underline,
            )
            screen.blit(surf, (rect.x + scroll_x, rect.y + scroll_y))

            if debug:
                r = pygame.Rect(rect.x + scroll_x, rect.y + scroll_y, rect.width, rect.height)
                pygame.draw.rect(screen, (255, 0, 0), r, 1)
                pygame.draw.rect(screen, (0, 255, 0), r.inflate(5, 5), 1)
                s = self.style.font_obj(20).render(repr(text), True, (255, 0, 0))
                screen.blit(s, (rect.x + scroll_x, rect.y + scroll_y))

    @property
    def indent_px(self):
        return self.style.font_size * self.indent

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


class NewLine(InlineText):
    def __init__(self, style: Style, **kwargs):
        super().__init__("", style, hard_break=True, **kwargs)


@dataclass
class Paragraph:
    text: str
    start: int
    end: int


class Document:

    def __init__(self, children: list[InlineText]) -> None:
        self.size = (0, 0)
        self.children = children

    def layout(self, width: float):
        write_head = (0, 0)
        indent = 0
        for child in self.children:
            indent += child.indent_px
            child.layout(indent, width - indent, write_head)
            write_head = child.continuation_pos

        self.size = max(child.rect.right for child in self.children), max(
            child.rect.bottom for child in self.children
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
        while i + 1 < len(self.children) and self.children[i + 1].is_before(x, y):
            i += 1

        return i, self.children[i]

    def paragraph_around(self, idx: int) -> Paragraph:
        # Find the start of the paragraph
        start = idx + 1
        while start > 0 and not self.children[start - 1].hard_break:
            start -= 1

        # Find the end of the paragraph
        end = idx
        while end + 1 < len(self.children) and not self.children[end + 1].hard_break:
            end += 1

        text = "".join(child.text for child in self.children[start : end + 1])
        return Paragraph(text, start, end + 1)

    def get(self, anchor: str) -> int | None:
        for i, child in enumerate(self.children):
            if child.anchor == anchor:
                return i
        return None


def flatten(node, style: Style) -> Generator[InlineText, None, None]:
    """Flatten a document tree into a list of InlineText objects."""

    if isinstance(node, marko.block.Document):
        for child in node.children:
            yield from flatten(child, style)
    elif isinstance(node, marko.block.Paragraph):
        for child in node.children:
            yield from flatten(child, style)
        yield NewLine(style)
    elif isinstance(node, marko.block.BlankLine):
        yield InlineText("\n\n", style, hard_break=True)
    elif isinstance(node, marko.block.Heading):
        style = style.with_class(f"h{node.level}")
        new = [n for child in node.children for n in flatten(child, style)]
        title = "".join(n.text for n in new)
        kebab = re.sub(r"\W+", "-", title.lower().strip())
        yield InlineText("", style, anchor=kebab)
        yield from new
        yield NewLine(style)
    elif isinstance(node, marko.block.List):
        for i, child in enumerate(node.children, start=node.start):
            if node.ordered:
                prefix = f"{i}. "
            else:
                prefix = node.bullet + " "
            yield InlineText(prefix, style, indent=1)
            yield from flatten(child, style)
            yield NewLine(style, indent=-1)
    elif isinstance(node, marko.block.ListItem):
        for child in node.children:
            yield from flatten(child, style)
    elif isinstance(node, marko.inline.RawText):
        yield InlineText(node.children, style)
    elif isinstance(node, marko.inline.LineBreak):
        if node.soft:
            yield InlineText(" ", style)
        else:
            yield InlineText("", style, hard_break=True)
    elif isinstance(node, marko.inline.Literal):
        assert isinstance(node.children, str)  # It should!
        yield InlineText(node.children, style)
    elif isinstance(node, marko.inline.Link):
        for child in node.children:
            for new in flatten(child, style + "link"):
                if not new.link_to:
                    new.link_to = node.dest
                yield new
    else:
        warnings.warn(f"Unsupported element: {node.__class__.__name__}")
        yield InlineText(f"<{node.__class__.__name__}> ", style + "error", hard_break=True)
        if hasattr(node, "children") and isinstance(node.children, list):
            for child in node.children:
                yield from flatten(child, style + "error")
        else:
            yield InlineText(repr(node), style + "error")
        yield NewLine(style)


def from_marko(doc, style: Style) -> Generator[InlineText, None, None]:
    # The main goal here is to filter redundant newlines.
    # We want to keep only one newline in a row.

    show_branches(doc)

    last = InlineText("dummy", style)
    for child in flatten(doc, style):
        if not last.text and not child.text:
            last.hard_break |= child.hard_break
            last.indent += child.indent
            continue
        yield child
        last = child


def show_branches(doc):
    counter = Counter()

    def recurse(node, path: str):
        path += node.__class__.__name__
        counter[path] += 1
        if isinstance(node.children, list):
            for child in node.children:
                recurse(child, path + ".")

    recurse(doc, "")
    pprint(counter)
