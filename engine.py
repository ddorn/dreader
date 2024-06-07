from __future__ import annotations

import dataclasses
import json
import re
import warnings
from dataclasses import dataclass
from pathlib import Path
from pprint import pprint
from typing import Counter, Generator

import marko.ast_renderer
import marko.block
import marko.inline
import pygame
import yaml

from dfont import TextParts, wrap, render
from style import Style, ComputedStyle, FontStyle

CSS = yaml.safe_load(Path("style.yaml").read_text())
SIZE_ATTRIBUTES = {"font_size"}
RE_SIZE = re.compile(r"([\d.]+)(\w+)")


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
        self.computed_style: ComputedStyle | None = None
        self.hard_break = hard_break
        self.link_to = link_to
        self.anchor = anchor
        self.indent = indent

        # Set by layout()
        self.size = (0, 0)
        self.text_parts: TextParts = None  # type: ignore
        self.continuation_pos = (0, 0)
        self.last_total_indent = 0

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

    def compute_style(self, base_size: int) -> ComputedStyle:
        if self.computed_style is None:
            self.computed_style = self.style.compute(base_size)
        return self.computed_style

    def add_class(self, cls: str):
        self.style += cls
        self.computed_style = None

    def remove_class(self, cls: str):
        self.style -= cls
        self.computed_style = None

    def layout(self, indent: float, width: float, type_head: tuple[int, int], base_size: int):
        style = self.compute_style(base_size)

        metrics = wrap(self.text, style.font.get_font(), int(width), type_head[0])
        metrics = metrics.shift(0, dy=type_head[1])

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
        self.last_total_indent = indent

    def render(self, scroll_x: int, scroll_y: int, screen):
        rect_on_screen = self.rect.move(scroll_x, scroll_y)
        if not rect_on_screen.colliderect(screen.get_rect()):
            return

        if self.computed_style is None:
            print(self)
        assert self.computed_style is not None, "layout() must be called before render()"

        # debug = pygame.key.get_pressed()[pygame.K_BACKSLASH]

        for text, rect in self.text_parts.parts:
            surf = render(text, self.computed_style.font)
            screen.blit(surf, (rect.x + scroll_x, rect.y + scroll_y))

            # if debug:
            #     r = pygame.Rect(rect.x + scroll_x, rect.y + scroll_y, rect.width, rect.height)
            #     pygame.draw.rect(screen, (255, 0, 0), r, 1)
            #     pygame.draw.rect(screen, (0, 255, 0), r.inflate(5, 5), 1)
            #     s = self.style.font_obj(20).render(repr(text), True, (255, 0, 0))
            #     screen.blit(s, (rect.x + scroll_x, rect.y + scroll_y))

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

    def layout(self, width: float, base_size: int):
        """Place and size all the elements in the document. Must be called before render()"""

        write_head = (0, 0)
        indent = 0
        for child in self.children:
            style = child.compute_style(base_size)
            indent += style.indent_size
            child.layout(indent, width - indent, write_head, base_size)
            write_head = child.continuation_pos

        self.size = max(child.rect.right for child in self.children), max(
            child.rect.bottom for child in self.children
        )

    def update_layout(
        self, width: float, base_size: int, scroll_x: float, scroll_y: float, screen_height: int
    ):
        """Update the layout of the visible elements.

        Accomodates changes to local changes such as font size, but not global
        ones such as indent (which can have arbitrary far-reaching effects).
        """

        # Find the first element that is curently visible
        i, node = self.at(-scroll_x, -scroll_y - 100)
        # Find first hard break before that
        while i > 0 and not self.children[i].hard_break:
            i -= 1

        # While we're not at the end of the screen, layout the elements
        if i == 0:
            write_head = (0, 0)
            indent = 0
        else:
            write_head = self.children[i - 1].continuation_pos
            indent = self.children[i - 1].last_total_indent

        while write_head[1] < -scroll_y + screen_height:
            if i >= len(self.children):
                break

            style = self.children[i].compute_style(base_size)
            indent += style.indent_size
            self.children[i].layout(indent, width - indent, write_head, base_size)
            write_head = self.children[i].continuation_pos

            i += 1

    def render(self, x, y, screen):
        for i, child in enumerate(self.children):
            try:
                child.render(x, y, screen)
            except AssertionError:
                print(f"Error rendering {i}: {child}")
                raise

    @classmethod
    def read(cls, path: Path, style: Style = Style([])):
        with open(path.expanduser()) as f:
            raw_text = f.read()

        docu = marko.parse(raw_text)

        d = marko.ast_renderer.ASTRenderer().render(docu)
        with open("out.json", "w") as f:
            json.dump(d, f, indent=2)

        return cls(list(from_marko(docu, style)))

    @classmethod
    def from_marko(cls, doc, style: Style = Style([])):
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
        style = style + "h" + f"h{node.level}"
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
            yield InlineText(prefix, style + "item")
            yield from flatten(child, style)
            yield NewLine(style + "item-end")
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
    elif isinstance(node, marko.inline.Emphasis):
        for child in node.children:
            yield from flatten(child, style + "italic")
    elif isinstance(node, marko.inline.StrongEmphasis):
        for child in node.children:
            yield from flatten(child, style + "bold")
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


def from_marko(doc, style: Style = Style([])) -> Generator[InlineText, None, None]:
    # The main goal here is to filter redundant newlines.
    # We want to keep only one newline in a row.

    show_branches(doc)

    stream = flatten(doc, style)
    last = next(stream)
    for child in stream:
        # We try to merge them into one
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
