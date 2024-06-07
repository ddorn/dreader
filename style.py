import types
import warnings
from dataclasses import dataclass, fields, is_dataclass
from pathlib import Path
from typing import Self, Type

import yaml

import dfont

CSS = yaml.safe_load(Path("style.yaml").read_text())
_SENTINEL = object()


@dataclass
class FontStyle:
    file: Path | str
    size: int
    color: tuple[int, int, int] = (0, 0, 0)
    background: tuple[int, int, int] | None = None
    underline: tuple[int, int, int] | None = None
    bold: bool = False
    italic: bool = False

    def get_font(self):
        return dfont.get_font(path=self.file, size=self.size, bold=self.bold, italic=self.italic)


@dataclass
class ComputedStyle:
    font: FontStyle
    indent_size: int


class Style:

    def __init__(self, classes: list[str]) -> None:
        # classes could be a set, BUT we need the order in which the styles are applied
        self.classes = classes

    def with_class(self, cls: str) -> Self:
        if cls not in self.classes:
            new_classes = self.classes + [cls]
        else:
            new_classes = list(self.classes)

        return self.__class__(new_classes)

    def without_class(self, cls: str) -> Self:
        new_classes = [c for c in self.classes if c != cls]
        return self.__class__(new_classes)

    __add__ = with_class
    __sub__ = without_class

    def compute[
        T
    ](
        self, base_size: int, path_in_css: tuple[str, ...] = (), schema: Type[T] = ComputedStyle
    ) -> T:

        assert is_dataclass(schema), "schema must be a dataclass"

        kwargs = {}
        for fld in fields(schema):
            fld_path_in_css = path_in_css + (fld.name,)
            if is_dataclass(fld.type):
                value = self.compute(base_size, fld_path_in_css, fld.type)
            else:
                value = self.recursive_get(CSS["default"], fld_path_in_css, _SENTINEL)
                assert value is not _SENTINEL, f"Default value not found for {fld_path_in_css}"

                for cls in self.classes:
                    if cls in CSS:
                        value = self.recursive_get(CSS[cls], fld_path_in_css, value)
                    else:
                        warnings.warn(f"Unknown class: {cls}")

                if fld.name.endswith("size"):
                    value = self.parse_size(value, base_size)  # type: ignore[arg-type]

            kwargs[fld.name] = value

        return schema(**kwargs)

    @staticmethod
    def recursive_get(d: dict, path, default=None):
        for key in path:
            if key in d:
                d = d[key]
            else:
                return default
        return d

    @staticmethod
    def parse_size(size: str | float, base_size: float) -> float:
        if isinstance(size, (float, int)):
            return size
        elif isinstance(size, str):
            if size.endswith("%"):
                return float(size[:-1]) / 100 * base_size
            elif size.endswith("rem"):
                return float(size[:-3]) * base_size
            else:
                raise ValueError(f"Invalid size: {size!r}")
        else:
            raise TypeError(f"Invalid size type: {type(size)}")


def _check_css(obj, schema=ComputedStyle, path=()) -> bool:
    path_str = ".".join(path)

    if is_dataclass(schema):
        possible_names = {fld.name for fld in fields(schema)}
        actual_names = set(obj.keys())
        for name in actual_names - possible_names:
            warnings.warn(f"Unknown field {name} at {'.'.join(path)}")

        return all(
            _check_css(obj[fld.name], fld.type, path + (fld.name,))
            for fld in fields(schema)
            if fld.name in obj
        )

    elif schema == tuple[int, int, int]:
        assert isinstance(obj, (tuple, list)), f"Expected tuple, got {type(obj)} at {path_str}"
        assert len(obj) == 3, f"Expected 3 elements, got {len(obj)} at {path_str}"
        assert all(isinstance(i, int) for i in obj), "Expected all ints at {path_str}"

    elif schema == (tuple[int, int, int] | None):
        assert obj is None or isinstance(
            obj, (tuple, list)
        ), f"Expected tuple or None, got {type(obj)} at {path_str}"
        if obj is not None:
            assert len(obj) == 3, f"Expected 3 elements, got {len(obj)} at {path_str}"
            assert all(isinstance(i, int) for i in obj), "Expected all ints at {path_str}"

    elif path[-1].endswith("size"):
        assert isinstance(
            obj, (float, int, str)
        ), f"Expected float or int, got {type(obj)} at {path_str}"
        if isinstance(obj, str):
            assert obj.endswith("%") or obj.endswith(
                "rem"
            ), f"Expected % or rem, got {obj} at {path_str}"

    else:
        assert isinstance(obj, schema), f"Expected {schema}, got {type(obj)} at {path_str}"


def check_css(css):
    assert isinstance(css, dict), "css must be a dict"
    for klass, value in css.items():
        assert isinstance(klass, str), "class names must be strings"
        _check_css(value, ComputedStyle, (klass,))


check_css(CSS)
