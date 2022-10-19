import open3d as o3d
import open3d.visualization.gui as gui

from .callback_manager import CallbackManager
from .geometry_manager import GeometryManager
from .ui_settings import UISettings
from .widget_manager import WidgetManager, WidgetType


# from .ui_settings import


class Window:
    def __init__(self, width=1200, height=900, name="PyRaT"):
        # basic info and modules
        gui.Application.instance.initialize()
        self.w, self.h, self.name = width, height, name
        self.entity = gui.Application.instance.create_window(self.name, self.w, self.h)
        self.em = self.entity.theme.font_size
        self.mgn_qtr = 0.25 * self.em
        self.mgn_half = 0.5 * self.em
        self.sep_h = int(round(self.mgn_half))
        # manager
        self.callback_mgr = CallbackManager(self)
        self.widget_mgr = WidgetManager(self)
        self.geometry_mgr = GeometryManager(self)
        self.settings = UISettings()
        # init base scene widget
        self.widget_mgr.add_scene("scene")
        # init ui
        self.__init_settings_panel()
        self.__init_menubar()
        # apply default settings
        self.apply_settings()

    def __init_settings_panel(self):
        self.settings_panel = gui.Vert(
            0, gui.Margins(self.mgn_half, self.mgn_half, self.mgn_half, self.mgn_half)
        )
        self.__view_controls()
        # self.settings_panel.add_child(self.__view_controls())
        # self.settings_panel.add_fixed(self.sep_h)
        self.settings_panel.add_child(self.__algorithm2d_controls())
        self.settings_panel.add_fixed(self.sep_h)

        self.entity.set_on_layout(self.__on_layout)
        self.entity.add_child(self.widget_mgr.widget(WidgetType.SCENE, "scene"))
        self.entity.add_child(self.settings_panel)

    def __init_menubar(self):
        if gui.Application.instance.menubar is None:
            window_menu = gui.Menu()
            self.widget_mgr.add_menu_item(
                window_menu, "quit", self.widget_mgr.menu_tags.MENU_QUIT
            )

            help_menu = gui.Menu()
            self.widget_mgr.add_menu_item(
                help_menu, "about", self.widget_mgr.menu_tags.MENU_ABOUT
            )

            menu = gui.Menu()
            menu.add_menu("Window", window_menu)
            menu.add_menu("Help", help_menu)

            gui.Application.instance.menubar = menu

    def apply_settings(self):
        # update scene
        img = o3d.io.read_image(
            "/home/jqlab/PycharmProjects/RAToolbox/test/util/img_1.png"
        )
        self.scene.scene.set_background(
            [
                self.settings.bg_color.red,
                self.settings.bg_color.green,
                self.settings.bg_color.blue,
                self.settings.bg_color.alpha,
            ],
            img,
        )
        self.scene.scene.scene.enable_indirect_light(self.settings.use_ibl)
        self.scene.scene.scene.set_indirect_light_intensity(self.settings.ibl_intensity)
        self.scene.scene.scene.enable_sun_light(self.settings.use_sun)
        if self.settings.apply_material:
            self.scene.scene.update_material(self.settings.material)
            self.settings.apply_material = False
        # update widgets
        self.widget_mgr.widget(
            WidgetType.COLOR_EDIT, "background_color"
        ).color_value = self.settings.bg_color
        c = gui.Color(
            self.settings.material.base_color[0],
            self.settings.material.base_color[1],
            self.settings.material.base_color[2],
            self.settings.material.base_color[3],
        )

    def __view_controls(self):
        # init widgets
        self.widget_mgr.add_button("arcball")
        self.widget_mgr.add_button("fly")
        self.widget_mgr.add_button("model")
        self.widget_mgr.add_label("mouse controls")
        self.widget_mgr.add_label("background color")
        self.widget_mgr.add_color_edit("background_color")

        # define layout
        view_controls = gui.CollapsableVert(
            "View Controls", self.mgn_qtr, gui.Margins(self.em, 0, 0, 0)
        )
        view_controls.add_child(
            self.widget_mgr.widget(WidgetType.LABEL, "mouse controls")
        )
        h = gui.Horiz(self.mgn_qtr)
        h.add_stretch()
        h.add_child(self.widget_mgr.widget(WidgetType.BUTTON, "arcball"))
        h.add_child(self.widget_mgr.widget(WidgetType.BUTTON, "fly"))
        h.add_child(self.widget_mgr.widget(WidgetType.BUTTON, "model"))
        h.add_stretch()
        view_controls.add_child(h)

        grid = gui.VGrid(2, self.mgn_qtr)
        grid.add_child(self.widget_mgr.widget(WidgetType.LABEL, "background color"))
        grid.add_child(
            self.widget_mgr.widget(WidgetType.COLOR_EDIT, "background_color")
        )
        view_controls.add_child(grid)

        return view_controls

    def __algorithm2d_controls(self):
        # init widgets
        self.widget_mgr.add_label("time end")
        self.widget_mgr.add_label("step")
        self.widget_mgr.add_label("tensor order")
        self.widget_mgr.add_label("taylor terms")
        self.widget_mgr.add_label("r0")
        self.widget_mgr.add_vector_edit("r0_edit")
        self.widget_mgr.add_number_edit("time end")
        self.widget_mgr.add_number_edit("step")
        self.widget_mgr.add_number_edit("tensor order")
        self.widget_mgr.add_number_edit("taylor terms")
        self.widget_mgr.add_button("run")

        # define layout
        algorithm2d_controls = gui.CollapsableVert(
            "ABS2008CDC", self.mgn_qtr, gui.Margins(self.em, 0, 0, 0)
        )

        h = gui.Horiz(self.mgn_qtr)
        h.add_child(self.widget_mgr.widget(WidgetType.LABEL, "time end"))
        h.add_stretch()
        h.add_child(self.widget_mgr.widget(WidgetType.NUMBER_EDIT, "time end"))
        algorithm2d_controls.add_child(h)

        h = gui.Horiz(self.mgn_qtr)
        h.add_child(self.widget_mgr.widget(WidgetType.LABEL, "step"))
        h.add_stretch()
        h.add_child(self.widget_mgr.widget(WidgetType.NUMBER_EDIT, "step"))
        algorithm2d_controls.add_child(h)

        h = gui.Horiz(self.mgn_qtr)
        h.add_child(self.widget_mgr.widget(WidgetType.LABEL, "tensor order"))
        h.add_stretch()
        h.add_child(self.widget_mgr.widget(WidgetType.NUMBER_EDIT, "tensor order"))
        algorithm2d_controls.add_child(h)

        h = gui.Horiz(self.mgn_qtr)
        h.add_child(self.widget_mgr.widget(WidgetType.LABEL, "taylor terms"))
        h.add_stretch()
        h.add_child(self.widget_mgr.widget(WidgetType.NUMBER_EDIT, "taylor terms"))
        algorithm2d_controls.add_child(h)

        h = gui.Horiz(self.mgn_qtr)
        h.add_stretch()
        h.add_child(self.widget_mgr.widget(WidgetType.LABEL, "r0"))
        h.add_child(self.widget_mgr.widget(WidgetType.VECTOR_EDIT, "r0_edit"))
        algorithm2d_controls.add_child(h)

        algorithm2d_controls.add_child(self.widget_mgr.widget(WidgetType.BUTTON, "run"))

        return algorithm2d_controls

    def __on_layout(self, layout_context):
        r = self.entity.content_rect
        self.scene.frame = r
        width = 17 * layout_context.theme.font_size
        height = min(
            r.height,
            self.settings_panel.calc_preferred_size(
                layout_context, gui.Widget.Constraints()
            ).height,
        )
        self.settings_panel.frame = gui.Rect(r.get_right() - width, r.y, width, height)

    @staticmethod
    def run():
        gui.Application.instance.run()

    @property
    def scene(self):
        return self.widget_mgr.widget(WidgetType.SCENE, "scene")
