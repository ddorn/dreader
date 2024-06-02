from __future__ import annotations

from abc import abstractmethod, ABC
from dataclasses import dataclass
from typing import Self
import warnings

import pygame
import marko.block
import marko.inline

from dfont import DFont


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

    @property
    def font(self):
        return self.font_obj(self.font_size)


class Element[T: Element](ABC):
    def __init__(self, children: list[T], style: Style):
        self.size = (0, 0)
        self.children = children
        self.style = style

    def layout(self):
        for child in self.children:
            child.layout()
        self._layout()

    @abstractmethod
    def _layout(self):
        raise NotImplementedError

    def render(self, x, y, screen):
        rect = pygame.Rect(x, y, *self.size)
        if not rect.colliderect(screen.get_rect()):
            return

        self._render(x, y, screen)

    @abstractmethod
    def _render(self, x, y, screen):
        raise NotImplementedError


class InlineText(Element):
    def __init__(self, text: str, style: Style):
        self.text = text
        super().__init__([], style)

    def _layout(self):
        self.size = self.style.font.size(self.text)

    def _render(self, x, y, screen):
        surf = self.style.font.render(self.text, True, (0, 0, 0))
        screen.blit(surf, (x, y))

    @classmethod
    def from_marko(cls, node: marko.inline.InlineElement, style: Style) -> InlineText:
        if isinstance(node, marko.inline.RawText):
            return InlineText(node.children, style)
        elif isinstance(node, marko.inline.LineBreak):
            if node.soft:
                return InlineText(" ", style)
            else:
                return InlineText("\n", style)
        else:
            warnings.warn(f"Unsupported inline element: {node.__class__.__name__}")
            return InlineText(
                f"Unsupported element: {node.__class__.__name__}",
                style.clone(color=(255, 0, 0)),
            )


class Paragraph(Element[InlineText]):
    def _layout(self):
        self.size = sum(child.size[0] for child in self.children), max(
            child.size[1] for child in self.children
        )

    def _render(self, x, y, screen):
        for child in self.children:
            child.render(x, y, screen)
            x += child.size[0]

    @classmethod
    def from_marko(cls, para: marko.block.BlockElement, style: Style) -> Paragraph:

        if isinstance(para, marko.block.Heading):
            style = style.clone(
                font_size=style.font_size * style.header_multipliers[para.level - 1]
            )

        children = []
        for child in para.children:
            if isinstance(child, marko.inline.InlineElement):
                children.append(InlineText.from_marko(child, style))
            else:
                warnings.warn(f"Unsupported element in Paragraph: {child.__class__.__name__}")
                children.append(
                    InlineText(
                        f"Unsupported element: {child.__class__.__name__}",
                        style.clone(color=(255, 0, 0)),
                    )
                )
        return Paragraph(children, style)


class Document(Element[Paragraph]):
    def _layout(self):
        self.size = max(child.size[0] for child in self.children), sum(
            child.size[1] for child in self.children
        )

    def _render(self, x, y, screen):
        for child in self.children:
            child.render(x, y, screen)
            y += child.size[1]

    @classmethod
    def from_marko(cls, doc: marko.block.Document, style: Style) -> Document:
        first_level = []
        for child in doc.children:
            if isinstance(child, (marko.block.Paragraph, marko.block.Heading)):
                first_level.append(Paragraph.from_marko(child, style))
            elif isinstance(child, marko.block.BlankLine):
                pass
            else:
                warnings.warn(f"Unsupported element in Document: {child.__class__.__name__}")
                first_level.append(
                    Paragraph(
                        [
                            InlineText(
                                f"Unsupported element: {child.__class__.__name__}",
                                style.clone(color=(255, 0, 0)),
                            )
                        ],
                        style,
                    )
                )
        return Document(first_level, style)
