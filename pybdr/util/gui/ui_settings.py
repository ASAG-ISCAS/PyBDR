import open3d as o3d
import open3d.visualization.gui as gui
import open3d.visualization.rendering as rendering


class UISettings:
    unlit = "defaultUnlit"
    lit = "defaultLit"
    normals = "normals"
    depth = "depth"

    material_names = ["lit", "unlit", "normals", "depth"]
    material_shaders = [lit, unlit, normals, depth]

    # lighting settings
    default_profile_name = "bright day wiht sun at +Y[default]"
    point_cloud_profile_name = "cloudy day(no direct sun)"
    custom_profile_name = "custom"
    lighting_profiles = {
        default_profile_name: {
            "ibl_intensity": 45000,
            "sun_intensity": 45000,
            "sun_dir": [0.577, -0.577, -0.577],
            # "ibl_rotation":
            "use_ibl": True,
            "use_sun": True,
        },
        "bright day with sun at -Y": {
            "ibl_intensity": 45000,
            "sun_intensity": 45000,
            "sun_dir": [0.577, 0.577, 0.577],
            # "ibl_rotation":
            "use_ibl": True,
            "use_sun": True,
        },
        "bright day with sun at +Z": {
            "ibl_intensity": 45000,
            "sun_intensity": 45000,
            "sun_dir": [0.577, 0.577, -0.577],
            # "ibl_rotation":
            "use_ibl": True,
            "use_sun": True,
        },
        "less bright day with sun at +Y": {
            "ibl_intensity": 35000,
            "sun_intensity": 50000,
            "sun_dir": [0.577, -0.577, -0.577],
            # "ibl_rotation":
            "use_ibl": True,
            "use_sun": True,
        },
        "less bright day with sun at -Y": {
            "ibl_intensity": 35000,
            "sun_intensity": 50000,
            "sun_dir": [0.577, 0.577, 0.577],
            # "ibl_rotation":
            "use_ibl": True,
            "use_sun": True,
        },
        "less bright day with sun at +Z": {
            "ibl_intensity": 35000,
            "sun_intensity": 50000,
            "sun_dir": [0.577, 0.577, -0.577],
            # "ibl_rotation":
            "use_ibl": True,
            "use_sun": True,
        },
        point_cloud_profile_name: {
            "ibl_intensity": 60000,
            "sun_intensity": 50000,
            "use_ibl": True,
            "use_sun": False,
            # "ibl_rotation":
        },
    }

    # material settings

    default_material_name = "polished ceramic"
    prefab = {
        default_material_name: {
            "metallic": 0.0,
            "roughness": 0.7,
            "reflectance": 0.5,
            "clearcoat": 0.2,
            "clearcoat_roughness": 0.2,
            "anisotropy": 0.0,
        },
        "metal (rougher)": {
            "metallic": 1.0,
            "roughness": 0.5,
            "reflectance": 0.9,
            "clearcoat": 0.0,
            "clearcoat_roughness": 0.0,
            "anisotropy": 0.0,
        },
        "metal (smoother)": {
            "metallic": 1.0,
            "roughness": 0.3,
            "reflectance": 0.9,
            "clearcoat": 0.0,
            "clearcoat_roughness": 0.0,
            "anisotropy": 0.0,
        },
        "plastic": {
            "metallic": 0.0,
            "roughness": 0.5,
            "clearcoat": 0.5,
            "clearcoat_roughness": 0.2,
            "anisotropy": 0.0,
        },
        ""
        " ceramic": {
            "metallic": 0.0,
            "roughness": 0.5,
            "reflectance": 0.9,
            "clearcoat": 1.0,
            "clearcoat_roughness": 0.1,
            "anisotropy": 0.0,
        },
        "clay": {
            "metallic": 0.0,
            "roughness": 1.0,
            "reflectance": 0.5,
            "clearcoat": 0.1,
            "clearcoat_roughness": 0.287,
            "anisotropy": 0.0,
        },
    }

    def __init__(self):
        self.mouse_mode = gui.SceneWidget.Controls.ROTATE_CAMERA
        self.bg_color = gui.Color(52 / 255, 52 / 255, 52 / 255)
        self.show_sky_box = False
        self.use_ibl = True
        self.use_sun = False
        self.new_ibl_name = None
        self.sun_intensity = 45000
        self.ibl_intensity = 45000
        self.sun_dir = [0, 0, 1]
        self.sun_color = gui.Color(1, 1, 1)
        self.apply_material = True
        # set material pool for future
        self._materials = {
            UISettings.unlit: rendering.MaterialRecord(),
            UISettings.lit: rendering.MaterialRecord(),
            UISettings.normals: rendering.MaterialRecord(),
            UISettings.depth: rendering.MaterialRecord(),
        }
        self._materials[UISettings.lit].base_color = [0.43, 0.43, 0.43, 1.0]
        self._materials[UISettings.lit].shader = UISettings.lit
        self._materials[UISettings.unlit].base_color = [0.43, 0.43, 0.43, 1.0]
        self._materials[UISettings.unlit].shader = UISettings.unlit
        self._materials[UISettings.normals].shader = UISettings.normals
        self._materials[UISettings.depth].shader = UISettings.depth

        self.material = self._materials[UISettings.lit]
        # define material for point cloud visualization
        self.pts_material = rendering.MaterialRecord()
        self.pts_material.point_size = 3
        self.pts_material.base_color = [1, 0, 0, 1]

    def set_material(self, name):
        self.material = self._materials[name]
        self.apply_material = True

    def apply_material_prefab(self, name):
        assert self.material.shader == UISettings.lit
        prefab = UISettings.prefab[name]
        for key, val in prefab.items():
            setattr(self.material, "base_" + key, val)

    def apply_lighting_profile(self, name):
        profile = UISettings.lighting_profiles[name]
        for key, val in profile.items():
            setattr(self, key, val)
