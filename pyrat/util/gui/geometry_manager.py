from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .window import Window


class GeometryManager:
    def __init__(self, window: "Window"):
        self._window = window
        self._objs = {}
