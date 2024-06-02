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


os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "hide"
os.environ["SDL_VIDEODRIVER"] = "x11"

import pygame
import pygame.locals as pg
import pygame._sdl2 as sdl2


from engine import Document, Style
from dfont import DFont

ROOT = Path(__file__).parent
ASSETS = ROOT / "assets"
FONT = ASSETS / "main.ttf"
CACHE = ROOT / "cache"
CACHE_INFO = CACHE / "info.jsonl"
CACHE.mkdir(exist_ok=True)


def color_from_name(name: str) -> tuple[int, int, int]:
    instance = random.Random(name)
    return instance.randint(0, 255), instance.randint(0, 255), instance.randint(0, 255)


def clamp(value, minimum, maximum):
    if value < minimum:
        return minimum
    if value > maximum:
        return maximum
    return value


def to_filename(name: str, max_len: int = 80) -> str:
    new = "".join(c if c.isalnum() else "_" for c in name[:max_len])
    return new.replace("__", "_").strip("_")


def split_long_paragraphs(text: str, sep=". ") -> list[str]:
    if len(text) > 4096:
        last_dot_before = text[:4096].rfind(sep)
        if last_dot_before == -1:
            last_dot_before = 4096
        else:
            last_dot_before += len(sep)
        text, rest = text[:last_dot_before], text[last_dot_before:]
        return [text] + split_long_paragraphs(rest)
    return [text]


P = ParamSpec("P")


def threaded(func: Callable[P, None]) -> Callable[P, threading.Thread]:
    def wrapper(*args, **kwargs) -> threading.Thread:
        thread = threading.Thread(target=func, args=args, kwargs=kwargs)
        thread.start()
        return thread

    return wrapper  # type: ignore


class TTS:
    def __init__(self, voice="alloy", speed: float = 1.0):
        self.voice = voice
        self.speed = speed
        self.current_loop = None

    @staticmethod
    def mk_path(text, voice, speed):
        base = f"{voice}_{speed}_{text}"
        h = hashlib.md5(base.encode()).hexdigest()
        return CACHE / (to_filename(f"{h}_{base}") + ".mp3")

    @classmethod
    async def create_completion(cls, text, voice, speed):
        path = cls.mk_path(text, voice, speed)
        if path.exists():
            print(f"Using cached TTS: {path}")
            return

        start = time.time()
        client = openai.AsyncClient()
        async with client.audio.speech.with_streaming_response.create(
            input=text,
            model="tts-1",
            voice=voice,
            response_format="mp3",
            speed=speed,
        ) as response:
            print(f"Downloading TTS: {path}")
            try:
                await response.stream_to_file(path)
            except Exception as e:
                path.unlink(missing_ok=True)
                raise e
        end = time.time()

        metadata = dict(
            path=path.name,
            text=text,
            voice=voice,
            speed=speed,
            time_to_generate=end - start,
            timestamp=time.time(),
        )
        pprint(metadata)

        with open(CACHE_INFO, "a") as f:
            f.write(json.dumps(metadata) + "\n")

    @staticmethod
    async def play_delayed(path, delay: float):
        for tries in range(30):
            await asyncio.sleep(delay)
            if path.exists():
                break
        else:
            raise FileNotFoundError(f"Cannot play audio: {path}")

        # We stream the audio to the file, so it might not be fully written yet.
        # Pygame will wait play only up to what's been generated.
        # So once pygame is done playing, we restart the audio from the end of what was played.
        pygame.mixer.music.load(path)
        pygame.mixer.music.play()

        start = time.time()
        while True:
            part_start = time.time()
            while pygame.mixer.music.get_busy():
                await asyncio.sleep(0.01)

            print(f"PLAY {time.time() - start:.3f} part: {time.time() - part_start:.3f}")
            # Just above the 10ms sleep - it did not have much to play -> probably the end
            if time.time() - part_start < 0.011:
                print("PLAY DONE")
                break

            pygame.mixer.music.load(path)
            pygame.mixer.music.play(start=time.time() - start)

    async def speak(self, text: str):

        parts = split_long_paragraphs(text)

        for part in parts:
            path = self.mk_path(part, self.voice, self.speed)
            await asyncio.gather(
                self.create_completion(part, self.voice, self.speed), self.play_delayed(path, 0.3)
            )

    async def highlight_spoken(self, layout, paragraph):
        # Wait for the audio to be started - but no more thn 3 seconds
        start = time.time()
        while not pygame.mixer.music.get_busy() and time.time() - start < 3:
            await asyncio.sleep(0.01)

        CHARS_PER_SECOND = 15.5
        start = time.time()
        while pygame.mixer.music.get_busy():
            time_passed = time.time() - start
            chars = int(time_passed * CHARS_PER_SECOND)

            # Find the first child that is after the point
            i = paragraph.start
            while i < paragraph.end:
                childs = layout.children[paragraph.start : i + 1]
                text = "".join(child.text for child in childs)
                if len(text) >= chars:
                    break
                i += 1

            if i == paragraph.end:
                i -= 1

            for child in layout.children[paragraph.start : i]:
                child.style -= "reading"
            layout.children[i].style += "reading"

            await asyncio.sleep(0.1)

        for child in layout.children[paragraph.start : paragraph.end]:
            child.style -= "reading"

        print("DONE HIGHLIGHTING", time.time() - start, paragraph)

    async def read_async(self, doc: Document, start: int = 0):
        while start < len(doc.children):
            paragraph = doc.paragraph_around(start)

            await asyncio.gather(
                self.speak(paragraph.text),
                self.highlight_spoken(doc, paragraph),
            )

            start = paragraph.end

    @threaded
    def read(self, doc: Document, start: int = 0):
        if self.current_loop is not None:
            self.stop()

        self.current_loop = asyncio.new_event_loop()
        try:
            self.current_loop.run_until_complete(self.read_async(doc, start))
        except RuntimeError:
            pass

    def stop(self):
        if self.current_loop is not None:
            self.current_loop.stop()
            while self.current_loop.is_running():
                time.sleep(0.01)
            self.current_loop.close()
            self.current_loop = None


SHOW_LOCALS = bool(os.getenv("SHOW_LOCALS", False))
app = typer.Typer(add_completion=False, pretty_exceptions_show_locals=SHOW_LOCALS)


@app.command()
def gui(
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

    tts = TTS()

    # %%
    with open(markdown.expanduser()) as f:
        raw_text = f.read()

    docu = marko.parse(raw_text)

    # d = marko.ast_renderer.ASTRenderer().render(docu)
    # with open("out.json", "w") as f:
    #     json.dump(d, f, indent=2)

    # %% Create the layout
    style = Style(main_font, base_size=font_size, color=text_color)
    layout = Document.from_marko(docu, style)

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
    layout.layout(window.size[0] * (1 - margin))
    max_doc_width = 900
    doc_width = min(max_doc_width, window.size[0] * (1 - margin))

    FPS = 60
    y_scroll = 50
    scroll_momentum = 0
    mouse_doc = (0, 0)

    hovered = None

    running = True
    while running:
        last_y_scroll = y_scroll
        for event in pygame.event.get():
            if event.type == pg.QUIT:
                running = False
            elif event.type == pg.WINDOWRESIZED:
                doc_width = min(max_doc_width, window.size[0] * (1 - margin))
                layout.layout(doc_width)
            elif event.type == pg.KEYDOWN:
                if event.key == pg.K_j:
                    scroll_momentum = -30
                elif event.key == pg.K_k:
                    scroll_momentum = +30
                elif event.key == pg.K_SPACE:
                    idx, _ = layout.at(*mouse_doc)
                    tts.read(layout, idx)
                elif event.key == pg.K_s:
                    tts.stop()
                elif event.key == pg.K_MINUS:
                    ...
                elif event.key == pg.K_PLUS or event.key == pg.K_EQUALS:
                    ...
            elif event.type == pg.MOUSEBUTTONDOWN:
                if event.button == 4:
                    scroll_momentum += 10
                elif event.button == 5:
                    scroll_momentum -= 10

        y_scroll += scroll_momentum * 60 / FPS
        scroll_momentum *= 0.8
        y_scroll = clamp(y_scroll, -layout.size[1] + screen.get_height() - 50, 50)
        x_scroll = (window.size[0] - doc_width) / 2

        mouse = pygame.mouse.get_pos()
        mouse_doc = mouse[0] - x_scroll, mouse[1] - y_scroll
        if hovered is not None:
            hovered.style -= "hovered"
        i, hovered = layout.at(*mouse_doc)
        hovered.style += "hovered"

        screen.fill(background_color)

        layout.render(x_scroll, y_scroll, screen)

        fps = clock.get_fps()
        fps_surf = main_font.render(f"{fps:.2f}", 20, (0, 0, 0))
        screen.fill((255, 255, 255), fps_surf.get_rect())
        screen.blit(fps_surf, (0, 0))

        debug_show(
            screen,
            y_scroll=y_scroll,
            busy=pygame.mixer.music.get_busy(),
        )

        window.flip()
        clock.tick(FPS)

    tts.stop()
    pygame.quit()


@app.command()
def download_tts(markdown: Path, voice: str = "alloy", speed: float = 1.0):
    with open(markdown.expanduser()) as f:
        raw_text = f.read()

    docu = marko.parse(raw_text)
    layout = Document.from_marko(docu, Style(DFont(FONT)))

    tts = TTS(voice, speed)

    texts = []
    i = 0
    while i < len(layout.children):
        paragraph = layout.paragraph_around(i)
        texts.extend(split_long_paragraphs(paragraph.text))
        i = paragraph.end

    async def dl():
        batchs = itertools.batched(texts, 50)
        last_time = 0
        for group in batchs:
            # Rate limit of 50 per minute
            if time.time() - last_time < 60:
                to_wait = 60 - (time.time() - last_time)
                print(f"Waiting {to_wait:.2f} seconds to avoid rate limit")
                await asyncio.sleep(to_wait)
            await asyncio.gather(*(tts.create_completion(text, voice, speed) for text in group))

    asyncio.run(dl())


if __name__ == "__main__":
    app()
