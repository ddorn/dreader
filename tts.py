import asyncio
import enum
import hashlib
import json
from pathlib import Path
from pprint import pprint
import subprocess
import threading
import time
from typing import Callable, ParamSpec
import warnings

import openai
import pygame

from engine import Document, InlineText

CACHE = Path("~/.cache/dreader").expanduser()
CACHE_INFO = CACHE / "info.jsonl"
CACHE.mkdir(exist_ok=True)


def to_filename(name: str, max_len: int = 80) -> str:
    new = "".join(c if c.isalnum() else "_" for c in name[:max_len])
    return new.replace("__", "_").strip("_")


def split_long_paragraphs(text: str, sep=". ") -> list[str]:
    text = text.strip()
    if not text:
        return []

    if len(text) > 4096:
        last_dot_before = text[:4096].rfind(sep)
        if last_dot_before == -1:
            last_dot_before = 4096
        else:
            last_dot_before += len(sep)
        text, rest = text[:last_dot_before], text[last_dot_before:]
        return [text.strip()] + split_long_paragraphs(rest)
    return [text]


P = ParamSpec("P")


def threaded(func: Callable[P, None]) -> Callable[P, threading.Thread]:
    def wrapper(*args, **kwargs) -> threading.Thread:
        thread = threading.Thread(target=func, args=args, kwargs=kwargs)
        thread.start()
        return thread

    return wrapper  # type: ignore


class GenerationStatus(enum.Enum):
    NOT_GENERATED = 0
    GENERATING = 1
    GENERATED = 2
    FAILED = 3


class PlayStatus(enum.Enum):
    NOT_PLAYING = 0
    PLAYING = 1
    DONE = 2


class TTS:
    def __init__(self, voice="alloy", speed: float = 1.0):
        self.voice = voice
        self.speed = speed
        self.current_loop = None
        self.currently_read: InlineText | None = None
        self.generated = GenerationStatus.NOT_GENERATED
        self.play_status = PlayStatus.NOT_PLAYING

    @staticmethod
    def mk_path(text, voice, speed):
        base = f"{voice}_{speed}_{text}"
        h = hashlib.md5(base.encode()).hexdigest()
        return CACHE / (to_filename(f"{h}_{base}") + ".mp3")

    async def create_completion(self, text, voice, speed):
        path = self.mk_path(text, voice, speed)
        if path.exists():
            print(f"Using cached TTS: {path}")
            self.generated = GenerationStatus.GENERATED
            return path

        start = time.time()
        client = openai.AsyncClient()
        print(f"Downloading TTS: {path}")
        async with client.audio.speech.with_streaming_response.create(
            input=text,
            model="tts-1",
            voice=voice,
            response_format="mp3",
            speed=speed,
        ) as response:
            self.generated = GenerationStatus.GENERATING
            try:
                await response.stream_to_file(path)
            except Exception as e:
                path.unlink(missing_ok=True)
                self.generated = GenerationStatus.FAILED
                raise e
        end = time.time()
        self.generated = GenerationStatus.GENERATED

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

        return path

    async def play_delayed(self, path, delay: float):
        while self.generated == GenerationStatus.NOT_GENERATED:
            await asyncio.sleep(delay)

        # We stream the audio to the file, so it might not be fully written yet.
        # Pygame will wait play only up to what's been generated.
        # So once pygame is done playing, we restart the audio from the end of what was played.
        pygame.mixer.music.load(path)
        pygame.mixer.music.play()

        started_once_full = self.generated == GenerationStatus.GENERATED
        start = time.time()
        while True:
            while pygame.mixer.music.get_busy():
                await asyncio.sleep(0.01)

            if started_once_full:
                self.play_status = PlayStatus.DONE
                return

            if self.generated == GenerationStatus.FAILED:
                print("Generation failed")
                self.play_status = PlayStatus.DONE
                break

            if self.generated == GenerationStatus.GENERATED:
                started_once_full = True

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
        while self.play_status == PlayStatus.NOT_PLAYING:
            await asyncio.sleep(0.1)

        CHARS_PER_SECOND = 16.5
        total_time: float | None = None
        start = time.time()
        while self.play_status == PlayStatus.PLAYING:
            time_passed = time.time() - start

            if total_time is None and self.generated == GenerationStatus.GENERATED:
                # Compute the true length of the audio using ffprobe
                cmd = f"ffprobe -i {self.mk_path(paragraph.text, self.voice, self.speed)} -show_entries format=duration -v quiet -of csv='p=0'"
                try:
                    total_time = float(subprocess.check_output(cmd, shell=True).strip())
                except subprocess.CalledProcessError:
                    warnings.warn("Failed to get audio duration, is ffmpeg installed?")
                    total_time = len(paragraph.text) / CHARS_PER_SECOND

            if total_time is None:
                chars = int(time_passed * CHARS_PER_SECOND)
            else:
                chars = int(time_passed * len(paragraph.text) / total_time)

            print("Highlighting", chars, paragraph.text)

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

            self.set_read(layout.children[i])

            await asyncio.sleep(0.1)

        self.set_read(None)
        print("DONE HIGHLIGHTING", time.time() - start, paragraph)

    def set_read(self, node: InlineText | None):
        if self.currently_read is not None:
            self.currently_read.remove_class("reading")
        if node is not None:
            node.add_class("reading")
        self.currently_read = node

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
            pygame.mixer.music.stop()
            self.current_loop.stop()
            self.set_read(None)
            while self.current_loop.is_running():
                time.sleep(0.01)
            self.current_loop.close()
            self.current_loop = None
            self.play_status = PlayStatus.NOT_PLAYING
            self.generated = GenerationStatus.NOT_GENERATED
