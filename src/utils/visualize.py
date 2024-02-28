from typing import *
from PIL.Image import Image as PILImage
from numpy import ndarray
from trimesh import Trimesh
from torch import Tensor
from src.data.threed_front_scene import Room
from src.data.threed_front import CachedRoom
from src.data.threed_future_dataset import ThreedFutureDataset

import os
import tempfile
import subprocess
import textwrap

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import trimesh
from imageio import imread
import torch


def get_textured_objects(
    bbox_params_t: ndarray,
    objects_dataset: ThreedFutureDataset,
    classes: List[str],
    obj_features: Optional[ndarray]=None,
    objfeat_type: Optional[str]=None,
    with_cls: bool=True,
    get_bbox_meshes: bool=False,
    verbose=True
) -> Tuple[List[Trimesh], List[Trimesh], List[str], List[ndarray], List[str]]:
    """For each one of the boxes, replace them with a trimesh object after transformation."""
    # For a batch of boxes, only process the first one
    if bbox_params_t.ndim == 3:
        bbox_params_t = bbox_params_t[0]

    trimesh_meshes, bbox_meshes, obj_classes, obj_sizes, obj_ids = [], [], [], [], []
    for j in range(bbox_params_t.shape[0]):
        if with_cls:
            if bbox_params_t[j, :-7].argmax(-1) >= len(classes):  # empty class
                if verbose:
                    print("{:2d} Empty object probability: {:.3f}".format(
                        j, bbox_params_t[j, :-7].max()
                    ))
                obj_classes.append(None)  # place holder
                obj_sizes.append(None)
                obj_ids.append(None)
                continue

            if verbose:
                print("{:2d} Class probability: {:.3f}; Category: {}".format(
                    j, bbox_params_t[j, :-7].max(), classes[bbox_params_t[j, :-7].argmax(-1)]
                ))
        else:
            if bbox_params_t[j, -8] > 0.5:  # empty class probability
                if verbose:
                    print("{:2d} Empty object probability: {:.3f}".format(
                        j, bbox_params_t[j, 0]
                    ))
                obj_classes.append(None)  # place holder
                obj_sizes.append(None)
                obj_ids.append(None)
                continue

        query_size = bbox_params_t[j, -4:-1]
        if with_cls:
            query_label = classes[bbox_params_t[j, :-7].argmax(-1)]
        else:
            query_label = None
        if obj_features is not None:
            query_feature = obj_features[j]
            furniture, select_gap = objects_dataset.get_closest_furniture_to_objfeat_and_size(query_label, query_size, query_feature, objfeat_type)
        else:
            furniture, select_gap = objects_dataset.get_closest_furniture_to_box(query_label, query_size)

        # print(furniture.model_uid, furniture.model_jid)  # get the retrieved model name
        obj_ids.append(furniture.model_jid)

        if verbose:
            if not with_cls:
                print("{:2d} Select gap: {:4f}; Object category: {}".format(
                    j, select_gap, furniture.label
                ))
            else:
                print("{:2d} Select gap: {:4f}".format(
                    j, select_gap
                ))

        obj_classes.append(furniture.label)
        obj_sizes.append(furniture.size)

        # Extract the predicted affine transformation to position the mesh
        translation = bbox_params_t[j, -7:-4]
        theta = bbox_params_t[j, -1]
        R = np.zeros((3, 3))
        R[0, 0] = np.cos(theta)
        R[0, 2] = -np.sin(theta)
        R[2, 0] = np.sin(theta)
        R[2, 2] = np.cos(theta)
        R[1, 1] = 1.

        if obj_features is not None:
            # Instead of using retrieved object scale, we use predicted size
            raw_bbox_vertices = np.load(furniture.path_to_bbox_vertices, mmap_mode="r")
            raw_size = np.array([
                np.sqrt(np.sum((raw_bbox_vertices[4]-raw_bbox_vertices[0])**2))/2,
                np.sqrt(np.sum((raw_bbox_vertices[2]-raw_bbox_vertices[0])**2))/2,
                np.sqrt(np.sum((raw_bbox_vertices[1]-raw_bbox_vertices[0])**2))/2
            ])
            scale = query_size / raw_size
        else:
            scale = furniture.scale

        # Create a trimesh object to save
        tr_mesh = trimesh.load(furniture.raw_model_path, force="mesh")
        tr_mesh.visual.material.image = Image.open(furniture.texture_image_path)
        tr_mesh.vertices *= scale
        tr_mesh.vertices -= (tr_mesh.bounds[0] + tr_mesh.bounds[1]) / 2.
        tr_mesh.vertices = tr_mesh.vertices.dot(R) + translation
        trimesh_meshes.append(tr_mesh)

        if get_bbox_meshes:
            # Create a trimesh bounding box to save; TODO: add color to the bbox
            bbox_mesh = trimesh.load("src/utils/template_bbox.ply", force="mesh")
            bbox_mesh.vertices = bbox_mesh.vertices * query_size
            bbox_mesh.vertices = bbox_mesh.vertices.dot(R) + translation
            bbox_meshes.append(bbox_mesh)

    if verbose and j == bbox_params_t.shape[1]-1:
        print()  # newline at the end

    assert bbox_params_t.shape[0] == len(obj_classes) == len(obj_sizes) >= len(trimesh_meshes)
    return trimesh_meshes, bbox_meshes, obj_classes, obj_sizes, obj_ids


def get_floor_plan(
    scene: Union[Room, CachedRoom],
    floor_textures: List[str],
    rectangle_floor=True,
    room_size: Optional[Union[Tensor, ndarray, List[float]]]=None,
    room_angle: Optional[float]=None
) -> Trimesh:
    """Get a trimesh object of the floor plan with a random texture."""
    vertices, faces = scene.floor_plan
    vertices = vertices - scene.floor_plan_centroid
    uv = np.copy(vertices[:, [0, 2]])
    uv -= uv.min(axis=0)
    uv /= 0.3  # repeat every 30cm
    texture = np.random.choice(floor_textures)

    if rectangle_floor:
        floor_sizes = room_size if room_size is not None else \
            ((np.max(vertices, axis=0) - np.min(vertices, axis=0)) / 2.)
        if len(floor_sizes) == 3:
            floor_4corners = np.array([
                [-floor_sizes[0], 0., -floor_sizes[2]],
                [-floor_sizes[0], 0.,  floor_sizes[2]],
                [ floor_sizes[0], 0.,  floor_sizes[2]],
                [ floor_sizes[0], 0., -floor_sizes[2]],
            ])
        elif len(floor_sizes) == 4:
            floor_4corners = np.array([
                [floor_sizes[0], 0., floor_sizes[1]],  # left bottom
                [floor_sizes[0], 0., floor_sizes[3]],  # left top
                [floor_sizes[2], 0., floor_sizes[3]],  # right top
                [floor_sizes[2], 0., floor_sizes[1]],  # right bottom
            ])
        else:
            raise ValueError(f"Invalid floor size: {floor_sizes}")
        vertices = floor_4corners
        faces = np.array([[0, 1, 2], [0, 2, 3]])

    if room_angle is not None:
        R = np.zeros((3, 3))
        R[0, 0] = np.cos(room_angle)
        R[0, 2] = -np.sin(room_angle)
        R[2, 0] = np.sin(room_angle)
        R[2, 2] = np.cos(room_angle)
        R[1, 1] = 1.
        vertices = vertices.dot(R)

    tr_floor = trimesh.Trimesh(np.copy(vertices), np.copy(faces), process=False)
    tr_floor.visual = trimesh.visual.TextureVisuals(
        uv=np.copy(uv),
        material=trimesh.visual.material.SimpleMaterial(image=Image.open(texture))
    )

    return tr_floor


def floor_plan_from_scene(
    scene: Union[Room, CachedRoom],
    path_to_floor_plan_textures: str,
    without_room_mask=False,
    rectangle_floor=True,
    room_size: Optional[Union[Tensor, ndarray, List[float]]]=None,
    room_angle: Optional[float]=None
) -> Tuple[Trimesh, Optional[Tensor]]:
    """Get a trimesh object for the floor plan and a layout mask for the scene."""
    if not without_room_mask:
        room_layout = torch.from_numpy(
            np.transpose(scene.room_mask[None, :, :, 0:1], (0, 3, 1, 2))
        )
    else:
        room_layout = None

    # Also get a renderable for the floor plan
    tr_floor = get_floor_plan(
        scene,
        [os.path.join(path_to_floor_plan_textures, fi)
            for fi in os.listdir(path_to_floor_plan_textures)],
        rectangle_floor, room_size, room_angle
    )

    return tr_floor, room_layout


def export_scene(
    output_dir: str,
    trimesh_meshes: List[Trimesh],
    bbox_meshes: Optional[List[Trimesh]]=None,
    names: Optional[List[str]]=None
) -> None:
    """Export the scene as a directory of `.obj`, `.mtl` and `.png` files."""
    if names is None:
        names = ["object_{:02d}.obj".format(i) for i in range(len(trimesh_meshes))]
    mtl_names = ["material_{:02d}".format(i) for i in range(len(trimesh_meshes))]

    if bbox_meshes is not None and len(bbox_meshes) > 0:
        for i, b in enumerate(bbox_meshes):
            b.export(os.path.join(output_dir, "bbox_{:02d}.obj".format(i)))

    for i, m in enumerate(trimesh_meshes):
        obj_out, tex_out = trimesh.exchange.obj.export_obj(m, return_texture=True)

        with open(os.path.join(output_dir, names[i]), "w") as f:
            f.write(
                obj_out.replace("material.mtl", mtl_names[i]+".mtl")\
                    .replace("material_0.mtl", mtl_names[i]+".mtl")
            )

        # No material and texture to rename
        if tex_out is None:
            continue

        mtl_key = next(k for k in tex_out.keys() if k.endswith(".mtl"))
        path_to_mtl_file = os.path.join(output_dir, mtl_names[i]+".mtl")
        with open(path_to_mtl_file, "wb") as f:
            f.write(
                tex_out[mtl_key].replace(b"material_0.png", (mtl_names[i]+".png").encode("ascii"))\
                    .replace(b"material_0.jpeg", (mtl_names[i]+".jpeg").encode("ascii"))
            )
        tex_key = next(k for k in tex_out.keys() if not k.endswith(".mtl"))
        tex_ext = os.path.splitext(tex_key)[1]
        path_to_tex_file = os.path.join(output_dir, mtl_names[i]+tex_ext)
        with open(path_to_tex_file, "wb") as f:
            f.write(tex_out[tex_key])


def blender_render_scene(
    scene_dir: str,
    output_dir: str,
    output_suffix="",
    *,
    engine="CYCLES",
    top_down_view=False,
    num_images=8,
    camera_dist=1.2,
    resolution_x=1024,
    resolution_y=1024,
    cycle_samples=32,
    verbose=False,
    timeout=15*60.,
):
    BLENDER_SCRIPT_PATH = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "blender_script.py"
    )

    args = [
        _blender_binary_path(),
        "-b", "-P", BLENDER_SCRIPT_PATH,
        "--",
        "--scene_dir", scene_dir,
        "--output_dir", output_dir,
        "--output_suffix", output_suffix,
        "--engine", engine,
        "--num_images", str(num_images),
        "--camera_dist", str(camera_dist),
        "--resolution_x", str(resolution_x),
        "--resolution_y", str(resolution_y),
        "--cycle_samples", str(cycle_samples),
    ]
    if top_down_view:
        args += ["--top_down_view"]

    # Execute the command
    if verbose:
        subprocess.check_call(args)
    else:
        try:
            _ = subprocess.check_output(args, stderr=subprocess.STDOUT, timeout=timeout)  # return stdout
        except subprocess.CalledProcessError as exc:
            raise RuntimeError(f"{exc}: {exc.output}") from exc


def draw_scene_graph(objs, triples, vocab=None, **kwargs):
    """
    Use GraphViz to draw a scene graph. If vocab is not passed then we assume
    that objs and triples are python lists containing strings for object and
    relationship names.

    Using this requires that GraphViz is installed. On Ubuntu 16.04 this is easy:
    sudo apt-get install graphviz
    """
    output_filename = kwargs.pop('output_filename', 'graph.png')
    orientation = kwargs.pop('orientation', 'V')
    edge_width = kwargs.pop('edge_width', 6)
    arrow_size = kwargs.pop('arrow_size', 1.5)
    binary_edge_weight = kwargs.pop('binary_edge_weight', 1.2)
    ignore_dummies = kwargs.pop('ignore_dummies', True)

    object_types = kwargs.pop('object_types', None)
    predicate_types = kwargs.pop('predicate_types', None)

    if orientation not in ['V', 'H']:
        raise ValueError('Invalid orientation "%s"' % orientation)
    rankdir = {'H': 'LR', 'V': 'TD'}[orientation]

    if vocab is not None:
        # Decode object and relationship names
        assert torch.is_tensor(objs)
        assert torch.is_tensor(triples)
        objs_list, triples_list = [], []
        for i in range(objs.size(0)):
            objs_list.append(vocab['object_idx_to_name'][objs[i].item()])
        for i in range(triples.size(0)):
            s = triples[i, 0].item()
            p = vocab['pred_idx_to_name'][triples[i, 1].item()]
            o = triples[i, 2].item()
            triples_list.append([s, p, o])
        objs, triples = objs_list, triples_list

    elif object_types is not None and predicate_types is not None:
        # Decode object and relationship names
        if not isinstance(objs, torch.Tensor):
            objs = torch.tensor(objs)
        if not isinstance(triples, torch.Tensor):
            triples = torch.tensor(triples)
        assert torch.is_tensor(objs)
        assert torch.is_tensor(triples)
        objs_list, triples_list = [], []
        for i in range(objs.size(0)):
            objs_list.append(object_types[objs[i].item()])
        for i in range(triples.size(0)):
            s = triples[i, 0].item()
            p = predicate_types[triples[i, 1].item()]
            o = triples[i, 2].item()
            triples_list.append([s, p, o])
        objs, triples = objs_list, triples_list

    # General setup, and style for object nodes
    lines = [
        'digraph{',
        'graph [size="5,3",ratio="compress",dpi="300",bgcolor="transparent"]',
        'rankdir=%s' % rankdir,
        'nodesep="0.5"',
        'ranksep="0.5"',
        'node [shape="box",style="rounded,filled",fontsize="48",color="none"]',
        'node [fillcolor="lightpink1"]',
    ]
    # Output nodes for objects
    for i, obj in enumerate(objs):
        if ignore_dummies and obj == '__image__':
            continue
        lines.append('%d [label="%s"]' % (i, obj))

    # Output relationships
    next_node_id = len(objs)
    lines.append('node [fillcolor="lightblue1"]')
    for s, p, o in triples:
        if ignore_dummies and p == '__in_image__':
            continue
        lines += [
            '%d [label="%s"]' % (next_node_id, p),
            '%d->%d [penwidth=%f,arrowsize=%f,weight=%f]' % (
            s, next_node_id, edge_width, arrow_size, binary_edge_weight),
            '%d->%d [penwidth=%f,arrowsize=%f,weight=%f]' % (
            next_node_id, o, edge_width, arrow_size, binary_edge_weight)
        ]
        next_node_id += 1
    lines.append('}')

    # Now it gets slightly hacky. Write the graphviz spec to a temporary
    # text file
    ff, dot_filename = tempfile.mkstemp()
    with open(dot_filename, 'w') as f:
        for line in lines:
            f.write('%s\n' % line)
    os.close(ff)

    # Shell out to invoke graphviz; this will save the resulting image to disk,
    # so we read it, delete it, then return it.
    output_format = os.path.splitext(output_filename)[1][1:]
    os.system('dot -T%s %s > %s' % (output_format, dot_filename, output_filename))
    os.remove(dot_filename)
    img = imread(output_filename)
    os.remove(output_filename)

    return img


def _blender_binary_path() -> str:
    path = os.getenv("BLENDER_PATH", None)
    if path is not None:
        return path

    if os.path.exists("blender/blender-3.3.1-linux-x64/blender"):
        return "blender/blender-3.3.1-linux-x64/blender"

    raise EnvironmentError(
        "To render 3D models, install Blender version 3.3.1 or higher and "
        "set the environment variable `BLENDER_PATH` to the path of the Blender executable."
    )


def add_title(
    pil_image: PILImage, title: str,
    font_size=16, font_fill: Tuple[int, int, int]=(0, 0, 0),
    font_path: Optional[str]=None,
    space_between_lines=4, line_width: Optional[int]=None
) -> PILImage:
    if font_path is None:
        font_path = "/usr/share/fonts/truetype/ubuntu/Ubuntu-BI.ttf"
    assert os.path.exists(font_path), f"Font file not found: {font_path}"

    font = ImageFont.truetype(font_path, font_size)
    image_draw = ImageDraw.Draw(pil_image)

    # Auto break the title into multiple lines
    line_width = line_width if line_width is not None \
        else int(pil_image.width / (font_size / 2))
    lines = textwrap.wrap(title, width=line_width)

    y_text = 0
    for line in lines:
        line_bbox = font.getbbox(line)
        line_width, line_height = line_bbox[2] - line_bbox[0],\
            line_bbox[3] - line_bbox[1]
        x_text = (pil_image.width - line_width) / 2.
        image_draw.text((x_text, y_text), line, font_fill, font)
        y_text += (line_height + space_between_lines)

    return pil_image


def make_gif(pil_images: List[PILImage], output_path: str, **kwargs):
    assert len(pil_images) > 0, "No image to make gif"
    assert output_path.endswith(".gif"), "Output path must be a gif file"

    # Convert RGBA to RGB with white background by default
    # RGBA to gif would result quantization artifacts
    convert_rgba_to_rgb = kwargs.pop("convert_rgba_to_rgb", True)
    background = kwargs.pop("rgba_background", (255, 255, 255))
    if convert_rgba_to_rgb:
        for i, image in enumerate(pil_images):
            if image.mode == "RGBA":
                image_rgb = Image.new("RGB", image.size, background)
                image_rgb.paste(image, mask=image.split()[3])
                pil_images[i] = image_rgb

    # Set the kwargs for gif saving
    duration = kwargs.pop("duration", 1000/10)  # in ms; fps = 1000 / duration
    loop = kwargs.pop("loop", 0)  # 0: loop forever; n: loop n times; None: no loop
    disposal = kwargs.pop("disposal", 2)  # 2: restore to background color
    save_kwargs = {
        "save_all": True,
        "append_images": pil_images[1:],
        "duration": duration,
        "loop": loop,
        "disposal": disposal,
        "palette": pil_images[0].getpalette()
    }

    pil_images[0].save(output_path, **save_kwargs)
