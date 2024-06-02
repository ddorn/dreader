#!/usr/bin/env python

"""
READER - A simple markdown reader in pygame.
Copyright (C) 2024  Diego Dorn

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""

# %%

from datetime import datetime
from dataclasses import dataclass
import dataclasses
from functools import lru_cache
import json
import os
import subprocess
import sys
from time import time
from typing import Annotated
from pathlib import Path
import re
import marko.ast_renderer
import marko.inline
import typer
from typer import Argument, Option
from enum import Enum
import random
import atexit
import marko
from pprint import pprint


os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "hide"
os.environ["SDL_VIDEODRIVER"] = "x11"

import pygame
import pygame.locals as pg
import pygame._sdl2 as sdl2


from engine import Document, Style, from_marko
from dfont import DFont

ASSETS = Path(__file__).parent / "assets"
FONT = ASSETS / "main.ttf"


def color_from_name(name: str) -> tuple[int, int, int]:
    instance = random.Random(name)
    return instance.randint(0, 255), instance.randint(0, 255), instance.randint(0, 255)


def clamp(value, minimum, maximum):
    if value < minimum:
        return minimum
    if value > maximum:
        return maximum
    return value


app = typer.Typer(add_completion=False, pretty_exceptions_show_locals=False)


@app.command()
def main(
    # fmt: off
    markdown: Annotated[Path, Argument(help="Path to the markdown file to read.")] = Path("~/Documents/five.md"),
    font: Annotated[Path, Option(help="Path to the font for all text.")] = FONT,
    background_color: tuple[int, int, int] = (245, 245, 245),
    text_color: tuple[int, int, int] = (30, 20, 10),
    # fmt: on
) -> None:

    pygame.init()
    pygame.mixer.init()
    pygame.key.set_repeat(300, 20)

    window = sdl2.Window("READER", borderless=True, always_on_top=True)
    window.get_surface().fill(background_color)
    window.flip()

    screen = window.get_surface()
    clock = pygame.time.Clock()
    main_font = DFont(font)
    font_size = 30

    # %%
    with open(markdown.expanduser()) as f:
        raw_text = f.read()

    docu = marko.parse(raw_text)
    docu
    # %%
    from collections import Counter

    counts = Counter()

    def count_types(node, parent=""):
        counts[parent + type(node).__name__] += 1
        try:
            children = node.children
        except AttributeError:
            pass
        else:
            for child in children:
                count_types(child, parent + type(node).__name__ + ".")

    count_types(docu)
    pprint(counts)

    # %% Create the layout
    style = Style(main_font, font_size, text_color)
    layout = Document.from_marko(docu, style)
    margin = 0.1
    layout.layout(window.size[0] * (1 - margin))

    print(layout.children[:10])

    # %%

    y_scroll = 50
    scroll_momentum = 0

    while True:
        for event in pygame.event.get():
            if event.type == pg.QUIT:
                sys.exit()
            elif event.type == pg.WINDOWRESIZED:
                layout.layout(window.size[0] * (1 - margin))
            elif event.type == pg.KEYDOWN:
                if event.key == pg.K_j:
                    scroll_momentum = -30
                elif event.key == pg.K_k:
                    scroll_momentum = +30
                elif event.key == pg.K_MINUS:
                    ...
                elif event.key == pg.K_PLUS or event.key == pg.K_EQUALS:
                    ...
            elif event.type == pg.MOUSEBUTTONDOWN:
                if event.button == 4:
                    scroll_momentum += 10
                elif event.button == 5:
                    scroll_momentum -= 10

        y_scroll += scroll_momentum
        scroll_momentum *= 0.8

        y_scroll = clamp(y_scroll, -layout.size[1] + screen.get_height() - 50, 50)

        screen.fill(background_color)

        layout.render(window.size[0] * margin / 2, y_scroll, screen)

        fps = clock.get_fps()
        fps_surf = main_font.render(f"{fps:.2f}", 20, (0, 0, 0))
        screen.fill((255, 255, 255), fps_surf.get_rect())
        screen.blit(fps_surf, (0, 0))

        window.flip()
        clock.tick(60)


if __name__ == "__main__":
    app()
