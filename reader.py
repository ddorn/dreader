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

import itertools
import subprocess
import warnings
import joblib
import asyncio
from functools import wraps
import hashlib
import json
import os
from pprint import pprint
import sys
import threading
import time
from typing import Annotated, Callable, ParamSpec
from pathlib import Path
import marko.ast_renderer
import marko.inline
import openai
import tqdm
import typer
from typer import Argument, Option
import random
import marko

from tts import TTS, split_long_paragraphs


os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "hide"
os.environ["SDL_VIDEODRIVER"] = "x11"

import pygame
import pygame.locals as pg
import pygame._sdl2 as sdl2


from dfont import render, render_simple
from engine import Document, Style, InlineText


ROOT = Path(__file__).parent
ASSETS = ROOT / "assets"
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


SHOW_LOCALS = bool(os.getenv("SHOW_LOCALS", False))
app = typer.Typer(add_completion=False, pretty_exceptions_show_locals=SHOW_LOCALS)


@app.command()
def gui(
    # fmt: off
    markdown: Annotated[Path, Argument(help="Path to the markdown file to read.")] = Path("~/prog/reader/sample.md"),
    background_color: tuple[int, int, int] = (245, 245, 245),
    # fmt: on
) -> None:

    pygame.init()
    pygame.mixer.init()
    pygame.key.set_repeat(300, 20)

    window = sdl2.Window(
        "READER", borderless=True, always_on_top=True, resizable=True, size=(800, 600)
    )

    screen = window.get_surface()
    clock = pygame.time.Clock()
    font_size = 30

    tts = TTS()

    doc = Document.read(markdown)

    for n in doc.children[:20]:
        print(n)

    def debug_show(screen, **kwargs):
        debug = pygame.key.get_pressed()[pygame.K_BACKSLASH]
        if not debug:
            return

        y = 100
        for key, value in kwargs.items():
            if isinstance(value, pygame.Rect):
                pygame.draw.rect(screen, (255, 0, 0), value, 1)
                s = main_font(20).render(
                    repr(value), True, (255, 0, 0), bgcolor=(255, 255, 255, 100)
                )
                screen.blit(s, (value.x, value.y))
            else:
                s = main_font(20).render(
                    f"{key}: {value}", True, (255, 0, 0), bgcolor=(255, 255, 255, 100)
                )
                screen.blit(s, (0, y))
                y += s.get_height()

    # %%
    margin = 0.1
    max_doc_width = 900
    doc_width = min(max_doc_width, window.size[0] * (1 - margin))

    doc.layout(doc_width, font_size)
    print("Layout done")

    FPS = 60
    y_scroll = 50
    scroll_momentum = 0
    mouse_doc = (0, 0)

    hovered = None
    follow_read = False

    print("Started")

    frame_time = 1
    running = True
    while running:
        frame_start = time.time()

        for event in pygame.event.get():
            if event.type == pg.QUIT:
                running = False
            elif event.type == pg.WINDOWRESIZED:
                doc_width = min(max_doc_width, window.size[0] * (1 - margin))
                doc.layout(doc_width, font_size)
            elif event.type == pg.KEYDOWN:
                if event.key == pg.K_j:
                    scroll_momentum = -30
                elif event.key == pg.K_k:
                    scroll_momentum = +30
                elif event.key == pg.K_SPACE:
                    idx, _ = doc.at(*mouse_doc)
                    tts.read(doc, idx)
                elif event.key == pg.K_s:
                    tts.stop()
                elif event.key == pg.K_g:
                    if event.mod & pg.KMOD_SHIFT:
                        y_scroll = float("-inf")
                    else:
                        y_scroll = float("+inf")
                elif event.key == pg.K_f:
                    follow_read = not follow_read
                elif event.key == pg.K_MINUS:
                    ...
                elif event.key == pg.K_PLUS or event.key == pg.K_EQUALS:
                    ...
                elif event.key == pg.K_d:
                    _, node = doc.at(*mouse_doc)
                    print(node)
                    print(node.computed_style)
            elif event.type == pg.MOUSEBUTTONDOWN:
                if event.button == 1:
                    idx, node = doc.at(*mouse_doc)
                    if node.link_to:
                        if node.link_to.startswith("#"):
                            if (target_idx := doc.get(node.link_to[1:])) is not None:
                                # scroll to target
                                y_scroll = -doc.children[target_idx].rect.y + 50
                            else:
                                warnings.warn(f"Could not find target: {node.link_to}")
                        else:
                            os.system(f"xdg-open {node.link_to}")
                    else:
                        tts.read(doc, idx)

                elif event.button == 4:
                    scroll_momentum += 10
                elif event.button == 5:
                    scroll_momentum -= 10

        y_scroll += scroll_momentum * 60 / FPS
        scroll_momentum *= 0.8
        y_scroll = clamp(y_scroll, -doc.size[1] + screen.get_height() - 50, 50)
        x_scroll = (window.size[0] - doc_width) / 2

        if follow_read and tts.currently_read is not None:
            # Make it center of the screen (y)
            y_scroll = -tts.currently_read.rect.centery + screen.get_height() // 2

        mouse = pygame.mouse.get_pos()
        mouse_doc = mouse[0] - x_scroll, mouse[1] - y_scroll
        if hovered is not None:
            hovered.remove_class("hovered")
        i, hovered = doc.at(*mouse_doc)
        hovered.add_class("hovered")

        screen.fill(background_color)

        # doc.update_layout(doc_width, font_size, x_scroll, y_scroll, window.size[1])
        doc.update_layout_simple(font_size)
        doc.render(x_scroll, y_scroll, screen)

        fps_surf = render_simple(f"{1/frame_time:.2f}", 20)
        screen.fill((255, 255, 255), fps_surf.get_rect())
        screen.blit(fps_surf, (0, 0))

        # debug_show(
        #     screen,
        #     y_scroll=y_scroll,
        #     busy=pygame.mixer.music.get_busy(),
        # )

        window.flip()
        frame_time = time.time() - frame_start
        clock.tick(FPS)

    tts.stop()
    pygame.quit()


@app.command()
def download_tts(markdown: Path, voice: str = "alloy", speed: float = 1.0, make_mp3: bool = False):
    layout = Document.read(markdown)

    tts = TTS(voice, speed)

    texts = []
    i = 0
    while i < len(layout.children):
        paragraph = layout.paragraph_around(i)
        texts.extend(split_long_paragraphs(paragraph.text))
        i = paragraph.end

    ordered_files = []
    to_do = []
    for text in texts:
        path = TTS.mk_path(text, voice, speed)
        if not path.exists():
            to_do.append(text)
        ordered_files.append(path)

    async def dl():
        batchs = list(itertools.batched(to_do, 50))
        last_time = 0
        for group in batchs:
            # Rate limit of 50 per minute
            if time.time() - last_time < 60:
                to_wait = 60 - (time.time() - last_time)
                print(f"Waiting {to_wait:.2f} seconds to avoid rate limit")
                await asyncio.sleep(to_wait)
            await asyncio.gather(*(tts.create_completion(text, voice, speed) for text in group))

    asyncio.run(dl())

    if make_mp3:
        out = markdown.with_suffix(".mp3")
        cmd = f"ffmpeg -i 'concat:{'|'.join(str(f) for f in ordered_files)}' -c copy {out}"
        subprocess.check_call(cmd, shell=True)
        print(f"Prepared storyteller audio: {out}")


if __name__ == "__main__":
    app()
