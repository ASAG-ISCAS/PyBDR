from typing import TYPE_CHECKING
from enum import IntEnum
import open3d.visualization.gui as gui
import open3d.visualization.rendering as rendering

if TYPE_CHECKING:
    from .window import Window


class WidgetType(IntEnum):
    BUTTON = 0
    COLOR_EDIT = 1
    CHECKBOX = 2
    MENU_ITEM = 3
    COMBOBOX = 4
    SLIDER = 5
    LABEL = 6
    SCENE = 7
    VECTOR_EDIT = 8


class MenuTag(IntEnum):
    MENU_OPEN = 1
    MENU_EXPORT = 2
    MENU_QUIT = 3
    MENU_SHOW_SETTINGS = 11
    MENU_ABOUT = 21


class WidgetManager:
    def __init__(self, window: "Window"):
        self.__window = window
        self._widgets = {}
        for widget_type in WidgetType:
            self._widgets[widget_type] = {}

        self._menu_tags = MenuTag

    def add_button(self, name, hp_em=0.5, vp_em=0):
        widget_type = WidgetType.BUTTON
        assert name not in self._widgets[widget_type]
        self._widgets[widget_type][name] = gui.Button(name)
        self._widgets[widget_type][name].horizontal_padding_em = hp_em
        self._widgets[widget_type][name].vertical_padding_em = vp_em
        self.__window.callback_mgr.bind_button_on_clicked_callback(
            self.widget(widget_type, name), name
        )

    def add_color_edit(self, name, default_color_value=gui.Color(0, 0, 0)):
        widget_type = WidgetType.COLOR_EDIT
        assert name not in self._widgets[widget_type]
        self._widgets[widget_type][name] = gui.ColorEdit()
        self._widgets[widget_type][name].color_value = default_color_value
        self.__window.callback_mgr.bind_color_edit_on_value_changed_callback(
            self.widget(widget_type, name), name
        )

    def add_checkbox(self, name, checked=False):
        widget_type = WidgetType.CHECKBOX
        assert name not in self._widgets[widget_type]
        self._widgets[widget_type][name] = gui.Checkbox(name)
        self._widgets[widget_type][name].checked = checked
        self.__window.callback_mgr.bind_checkbox_on_checked_callback(
            self.widget(widget_type, name), name
        )

    def add_menu_item(self, menu, name, tag):
        menu.add_item(name, tag)
        self.__window.callback_mgr.bind_menu_item_activated_callback(name, tag)

    def add_label(self, name):
        widget_type = WidgetType.LABEL
        assert name not in self._widgets[widget_type]
        self._widgets[widget_type][name] = gui.Label(name)

    def add_combobox(self, name, items=None, default_selected_text=None):
        widget_type = WidgetType.COMBOBOX
        assert name not in self._widgets[widget_type]
        self._widgets[widget_type][name] = gui.Combobox()
        if items is None:
            return
        for item in items:
            self._widgets[widget_type][name].add_item(item)
        self.__window.callback_mgr.bind_combobox_on_selection_changed_callback(
            self.widget(widget_type, name), name
        )
        if default_selected_text is None:
            return
        self._widgets[widget_type][name].select_text = default_selected_text

    def add_scene(self, name):
        widget_type = WidgetType.SCENE
        assert name not in self._widgets[widget_type]
        self._widgets[widget_type][name] = gui.SceneWidget()
        self._widgets[widget_type][name].scene = rendering.Open3DScene(
            self.__window.entity.renderer
        )
        self.__window.callback_mgr.bind_scene_on_sun_direction_changed_callback(
            self.widget(widget_type, name), name
        )

    def add_slider(self, name, val_type=gui.Slider.DOUBLE, min_l=0, max_l=10, step=1):
        widget_type = WidgetType.SLIDER
        assert name not in self._widgets[widget_type] and (
            val_type == gui.Slider.INT or val_type == gui.Slider.DOUBLE
        )
        self._widgets[widget_type][name] = gui.Slider(val_type)
        self._widgets[widget_type][name].set_limits(min_l, max_l)
        if val_type is gui.Slider.INT:
            self._widgets[widget_type][name].int_value = step
        else:
            self._widgets[widget_type][name].double_value = step
        self.__window.callback_mgr.bind_slider_on_value_changed_callback(
            self.widget(widget_type, name), name
        )

    def add_vector_edit(self, name, default_vector_value=None):
        default_vector_value = (
            [0, 0, 1] if default_vector_value is None else default_vector_value
        )
        widget_type = WidgetType.VECTOR_EDIT
        assert name not in self._widgets[widget_type]
        self._widgets[widget_type][name] = gui.VectorEdit()
        self._widgets[widget_type][name].vector_value = default_vector_value
        self.__window.callback_mgr.bind_vector_edit_on_value_changed_callback(
            self.widget(widget_type, name), name
        )

    def widget(self, widget_type, name):
        assert name in self._widgets[widget_type]
        return self._widgets[widget_type][name]

    @property
    def menu_tags(self):
        return self._menu_tags
