# import open3d as o3d
# import open3d.visualization.gui as gui
import numpy as np
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pyrat.util.gui.window import Window


class CallbackManager:
    def __init__(self, window: "Window"):
        self._window = window
        self._callbacks = {}

    # ================================================================ binding functions

    def bind_button_on_clicked_callback(self, button, callback_name):
        on_clicked_callback_name = "_on_clicked_" + callback_name
        assert hasattr(self, on_clicked_callback_name)
        on_clicked_callback = getattr(self, on_clicked_callback_name)
        button.set_on_clicked(on_clicked_callback)

    def bind_color_edit_on_value_changed_callback(self, color_edit, callback_name):
        on_value_changed_callback_name = "_on_value_changed_" + callback_name
        assert hasattr(self, on_value_changed_callback_name)
        on_value_changed_callback = getattr(self, on_value_changed_callback_name)
        color_edit.set_on_value_changed(on_value_changed_callback)

    def bind_scene_on_sun_direction_changed_callback(self, scene, callback_name):
        on_sun_direction_changed_callback_name = (
            "_on_sun_direction_changed_" + callback_name
        )
        assert hasattr(self, on_sun_direction_changed_callback_name)
        on_sun_direction_changed_callback = getattr(
            self, on_sun_direction_changed_callback_name
        )
        scene.set_on_sun_direction_changed(on_sun_direction_changed_callback)

    def bind_menu_item_activated_callback(self, callback_name, tag):
        on_menu_item_activated_callback_name = (
            "_on_menu_item_activated_" + callback_name
        )
        assert hasattr(self, on_menu_item_activated_callback_name)
        on_menu_item_activated_callback = getattr(
            self, on_menu_item_activated_callback_name
        )
        self._window.entity.set_on_menu_item_activated(
            tag, on_menu_item_activated_callback
        )

    def bind_vector_edit_on_value_changed_callback(self, callback_name, vector_value):
        pass

    # =============================================================== callback functions

    # button callbacks

    def _on_clicked_run(self):
        pass

    def _on_clicked_arcball(self):
        self._window.scene.set_view_control(gui.SceneWidget.Controls.ROTATE_CAMERA)

    def _on_clicked_fly(self):
        self._window.scene.set_view_controls(gui.SceneWidget.Controls.FLY)

    def _on_clicked_model(self):
        self._window.scene.set_view_controls(gui.SceneWidget.Controls.ROTATE_MODEL)

    def _on_clicked_directional_vector(self):
        self._window.scene.set_view_controls(gui.SceneWidget.Controls.ROTATE_SUN)

    def _on_sun_direction_changed_scene(self, sun_dir):
        self._window.settings.sun_dir = sun_dir
        self._window.apply_settings()

    def _on_value_changed_background_color(self, color):
        self._window.settings.bg_color = color
        self._window.apply_settings()

    @staticmethod
    def _on_menu_item_activated_quit():
        gui.Application.instance.quit()

    @staticmethod
    def _on_menu_item_activated_about():
        print("about")
