"""Blender script to render images of 3D scenes composed of multiple objects.

Example usage:
    blender -b -P blender_script.py -- \
        --scene_dir outputs/scene_000 \
        --output_dir renderings \
        --engine CYCLES \
        --num_images 8 \
        --camera_dist 1.2
"""

import argparse
import math
import os
import sys
import time

import bpy
from mathutils import Vector


parser = argparse.ArgumentParser()
parser.add_argument("--scene_dir", type=str, required=True)
parser.add_argument("--output_dir", type=str, default="renderings")
parser.add_argument("--output_suffix", type=str, default="")
parser.add_argument("--engine", type=str, default="CYCLES", choices=["CYCLES", "BLENDER_EEVEE"])
parser.add_argument("--top_down_view", action="store_true")
parser.add_argument("--num_images", type=int, default=8)
parser.add_argument("--camera_dist", type=float, default=1.2)
parser.add_argument("--resolution_x", type=int, default=256)
parser.add_argument("--resolution_y", type=int, default=256)
parser.add_argument("--cycle_samples", type=int, default=32)

argv = sys.argv[sys.argv.index("--") + 1 :]
args = parser.parse_args(argv)

context = bpy.context
scene = context.scene
render = scene.render

render.engine = args.engine
render.image_settings.file_format = "PNG"
render.image_settings.color_mode = "RGB"
render.image_settings.color_depth = "8"
render.resolution_x = args.resolution_x
render.resolution_y = args.resolution_y
render.resolution_percentage = 100
render.film_transparent = False  # if you want transparent background

scene.cycles.device = "CPU"
scene.cycles.samples = args.cycle_samples  # number of samples per pixel
scene.cycles.diffuse_bounces = 1
scene.cycles.glossy_bounces = 1
scene.cycles.transparent_max_bounces = 3
scene.cycles.transmission_bounces = 3
scene.cycles.filter_width = 0.1  # blur with higher values, aliased with lower values
scene.cycles.use_denoising = True


def clear_lights():
    bpy.ops.object.select_all(action="DESELECT")
    for obj in bpy.context.scene.objects.values():
        if isinstance(obj.data, bpy.types.Light):
            obj.select_set(True)
    bpy.ops.object.delete()


def create_light(location, energy=1.):
    light_data = bpy.data.lights.new(name="Light", type="POINT")
    light_data.energy = energy
    light_object = bpy.data.objects.new(name="Light", object_data=light_data)

    bpy.context.collection.objects.link(light_object)
    light_object.location = location


def create_all_lights(num_lights=8, distance=2., energy=50.):
    clear_lights()

    for i in range(num_lights):
        theta = (i / num_lights) * math.pi * 2  # azimuth
        phi = math.radians(60)  # elevation
        locations = (
            math.sin(phi) * math.cos(theta),
            math.sin(phi) * math.sin(theta),
            math.cos(phi),
        )
        create_light(Vector(locations) * distance, energy=energy)


def reset_scene() -> None:
    """Resets the scene to a clean state."""
    # Delete everything that isn't part of a camera or a light
    for obj in bpy.data.objects:
        if obj.type not in {"CAMERA", "LIGHT"}:
            bpy.data.objects.remove(obj, do_unlink=True)
    # Delete all the materials
    for material in bpy.data.materials:
        bpy.data.materials.remove(material, do_unlink=True)
    # Delete all the textures
    for texture in bpy.data.textures:
        bpy.data.textures.remove(texture, do_unlink=True)
    # Delete all the images
    for image in bpy.data.images:
        bpy.data.images.remove(image, do_unlink=True)


def load_object(object_path: str) -> None:
    """Loads a `.obj` model into the scene."""
    if object_path.endswith(".obj"):
        bpy.ops.import_scene.obj(filepath=object_path)
    else:
        raise ValueError(f"Unsupported file type: {object_path}")


def load_scene_objects(scene_dir: str) -> None:
    """Loads all `.obj` models in a scene directory."""
    for object_path in os.listdir(scene_dir):
        if object_path.endswith(".obj"):
            load_object(os.path.join(scene_dir, object_path))


def scene_bbox(single_obj=None, ignore_matrix=False):
    bbox_min = (math.inf,) * 3
    bbox_max = (-math.inf,) * 3
    found = False
    for obj in scene_meshes() if single_obj is None else [single_obj]:
        found = True
        for coord in obj.bound_box:
            coord = Vector(coord)
            if not ignore_matrix:
                coord = obj.matrix_world @ coord
            bbox_min = tuple(min(x, y) for x, y in zip(bbox_min, coord))
            bbox_max = tuple(max(x, y) for x, y in zip(bbox_max, coord))
    if not found:
        raise RuntimeError("no objects in scene to compute bounding box for")
    return Vector(bbox_min), Vector(bbox_max)


def scene_root_objects():
    for obj in bpy.context.scene.objects.values():
        if not obj.parent:
            yield obj


def scene_meshes():
    for obj in bpy.context.scene.objects.values():
        if isinstance(obj.data, (bpy.types.Mesh)):
            yield obj


def normalize_scene():
    bbox_min, bbox_max = scene_bbox()
    scale = 1 / max(bbox_max - bbox_min)
    for obj in scene_root_objects():
        obj.scale = obj.scale * scale
    # Apply scale to matrix_world
    bpy.context.view_layer.update()
    bbox_min, bbox_max = scene_bbox()
    offset = -(bbox_min + bbox_max) / 2
    for obj in scene_root_objects():
        obj.matrix_world.translation += offset
    bpy.ops.object.select_all(action="DESELECT")


def setup_camera():
    cam = scene.objects["Camera"]
    cam.location = (0, 1.2, 0)
    cam.data.lens = 35
    cam.data.sensor_width = 32
    cam_constraint = cam.constraints.new(type="TRACK_TO")
    cam_constraint.track_axis = "TRACK_NEGATIVE_Z"
    cam_constraint.up_axis = "UP_Y"
    return cam, cam_constraint


def save_images(scene_dir: str) -> None:
    """Saves rendered images of a scene."""
    os.makedirs(args.output_dir, exist_ok=True)
    reset_scene()
    # Load a scene with all the objects
    load_scene_objects(scene_dir)

    normalize_scene()

    # Create the lights and camera
    create_all_lights()
    cam, cam_constraint = setup_camera()
    # Create an empty object to track
    empty = bpy.data.objects.new("Empty", None)
    scene.collection.objects.link(empty)
    cam_constraint.target = empty

    if args.top_down_view:
        # Render the image from a top-down view
        point = (0., 0., args.camera_dist)
        cam.location = point
        # Render the image
        render_path = os.path.join(args.output_dir, f"topdown{args.output_suffix}.png")
        scene.render.filepath = render_path
        bpy.ops.render.render(write_still=True)
    else:
        # Render the images from different angles
        for i in range(args.num_images):
            # Set the camera position
            theta = (i / args.num_images) * math.pi * 2  # azimuth
            phi = math.radians(60)  # elevation
            point = (
                args.camera_dist * math.sin(phi) * math.cos(theta),
                args.camera_dist * math.sin(phi) * math.sin(theta),
                args.camera_dist * math.cos(phi),
            )
            cam.location = point
            # Render the image
            render_path = os.path.join(args.output_dir, f"{i:03d}{args.output_suffix}.png")
            scene.render.filepath = render_path
            bpy.ops.render.render(write_still=True)


if __name__ == "__main__":
    start_i = time.time()
    save_images(args.scene_dir)
    end_i = time.time()

    print(f">>> Rendering scene from {args.scene_dir} in [{round(end_i - start_i, 2)}] seconds")
