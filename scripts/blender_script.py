"""Blender script for MVD-Fusion dataset preparation

This script renders RGB, Depth, and Mask images for 3D models in MVD-Fusion format.

Example usage:
    blender -b -P blender_script_mvd_fusion.py -- \
        --object_path my_object.glb \
        --output_dir ./views \
        --num_images 64
"""

import argparse
import math
import os
import sys
import time
import urllib.request
from typing import Tuple
import json
import bpy
from mathutils import Vector
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

parser = argparse.ArgumentParser()
parser.add_argument(
    "--object_path",
    type=str,
    required=True,
    help="Path to the object file",
)
parser.add_argument("--output_dir", type=str, default="./views")
parser.add_argument(
    "--engine", type=str, default="CYCLES", choices=["CYCLES", "BLENDER_EEVEE"]
)
parser.add_argument("--num_images", type=int, default=64)
parser.add_argument("--camera_dist", type=float, default=1.5)

argv = sys.argv[sys.argv.index("--") + 1 :]
args = parser.parse_args(argv)

context = bpy.context
scene = context.scene
render = scene.render

# MVD-Fusion render settings
render.engine = args.engine
render.image_settings.file_format = "JPEG"  # FIXED: JPG for RGB
render.image_settings.color_mode = "RGB"
render.image_settings.quality = 90
render.resolution_x = 256  # FIXED: MVD-Fusion resolution
render.resolution_y = 256
render.resolution_percentage = 100

scene.cycles.device = "GPU"
scene.cycles.samples = 32
scene.cycles.diffuse_bounces = 1
scene.cycles.glossy_bounces = 1
scene.cycles.transparent_max_bounces = 3
scene.cycles.transmission_bounces = 3
scene.cycles.filter_width = 0.01
scene.cycles.use_denoising = True
scene.render.film_transparent = True

# MVD-Fusion camera parameters (from dataset/objaverse.py)
AZIMUTHS_16 = [
    0.0, 0.7853981852531433, 1.5707963705062866, 2.356194496154785,
    3.1415927410125732, 3.9269907474517822, 4.71238899230957, 5.497786998748779,
    0.39269909262657166, 1.1780972480773926, 1.9634954929351807, 2.7488934993743896,
    3.5342917442321777, 4.319689750671387, 5.105088233947754, 5.890486240386963
]

ELEVATIONS_16 = [
    0.5235987901687622, 0.5235987901687622, 0.5235987901687622, 0.5235987901687622,
    0.5235987901687622, 0.5235987901687622, 0.5235987901687622, 0.5235987901687622,
    0.5235987901687622, 0.5235987901687622, 0.5235987901687622, 0.5235987901687622,
    0.5235987901687622, 0.5235987901687622, 0.5235987901687622, 0.5235987901687622
]


def setup_compositor_for_depth_and_mask(views_dir):
    """
    NEW: Setup compositor to render RGB, Depth, and Mask in single pass
    """
    scene = bpy.context.scene
    scene.use_nodes = True
    scene.view_layers["ViewLayer"].use_pass_z = True  # Enable depth
    
    tree = scene.node_tree
    tree.nodes.clear()
    
    # Render layers node
    render_layers = tree.nodes.new(type='CompositorNodeRLayers')
    render_layers.location = 0, 0
    
    # RGB output (goes to scene.render.filepath)
    composite_rgb = tree.nodes.new(type='CompositorNodeComposite')
    composite_rgb.location = 400, 200
    tree.links.new(render_layers.outputs['Image'], composite_rgb.inputs['Image'])
    
    # Depth normalization
    normalize = tree.nodes.new(type='CompositorNodeNormalize')
    normalize.location = 200, -100
    tree.links.new(render_layers.outputs['Depth'], normalize.inputs[0])
    
    # Map to 0-1 range
    map_range = tree.nodes.new(type='CompositorNodeMapValue')
    map_range.location = 400, -100
    map_range.offset = [0.0]
    map_range.size = [1.0]
    map_range.use_min = True
    map_range.min = [0.0]
    map_range.use_max = True
    map_range.max = [1.0]
    tree.links.new(normalize.outputs[0], map_range.inputs[0])
    
    # File output for depth
    depth_output = tree.nodes.new(type='CompositorNodeOutputFile')
    depth_output.location = 600, -100
    depth_output.base_path = views_dir
    depth_output.format.file_format = 'PNG'
    depth_output.format.color_mode = 'BW'
    depth_output.format.color_depth = '16'
    tree.links.new(map_range.outputs[0], depth_output.inputs[0])
    
    # Mask output (alpha channel)
    mask_output = tree.nodes.new(type='CompositorNodeOutputFile')
    mask_output.location = 600, -300
    mask_output.base_path = views_dir
    mask_output.format.file_format = 'JPEG'
    mask_output.format.color_mode = 'BW'
    mask_output.format.quality = 90
    tree.links.new(render_layers.outputs['Alpha'], mask_output.inputs[0])
    
    return depth_output, mask_output


def add_lighting() -> None:
    # Delete default light
    bpy.data.objects["Light"].select_set(True)
    bpy.ops.object.delete()
    # Add area light
    bpy.ops.object.light_add(type="AREA")
    light2 = bpy.data.lights["Area"]
    light2.energy = 30000
    bpy.data.objects["Area"].location[2] = 0.5
    bpy.data.objects["Area"].scale[0] = 100
    bpy.data.objects["Area"].scale[1] = 100
    bpy.data.objects["Area"].scale[2] = 100


def reset_scene() -> None:
    """Resets the scene to a clean state."""
    for obj in bpy.data.objects:
        if obj.type not in {"CAMERA", "LIGHT"}:
            bpy.data.objects.remove(obj, do_unlink=True)
    for material in bpy.data.materials:
        bpy.data.materials.remove(material, do_unlink=True)
    for texture in bpy.data.textures:
        bpy.data.textures.remove(texture, do_unlink=True)
    for image in bpy.data.images:
        bpy.data.images.remove(image, do_unlink=True)


def load_object(object_path: str) -> None:
    """Loads a glb model into the scene."""
    if object_path.endswith(".glb"):
        bpy.ops.import_scene.gltf(filepath=object_path, merge_vertices=True)
    elif object_path.endswith(".fbx"):
        bpy.ops.import_scene.fbx(filepath=object_path)
    else:
        raise ValueError(f"Unsupported file type: {object_path}")


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
    bpy.context.view_layer.update()
    bbox_min, bbox_max = scene_bbox()
    offset = -(bbox_min + bbox_max) / 2
    for obj in scene_root_objects():
        obj.matrix_world.translation += offset
    bpy.ops.object.select_all(action="DESELECT")


def setup_camera():
    cam = scene.objects["Camera"]
    cam.data.lens = 35
    cam.data.sensor_width = 32
    cam_constraint = cam.constraints.new(type="TRACK_TO")
    cam_constraint.track_axis = "TRACK_NEGATIVE_Z"
    cam_constraint.up_axis = "UP_Y"
    return cam, cam_constraint


def get_camera_position(azimuth, elevation, distance=1.5):
    """Get camera position from azimuth and elevation (MVD-Fusion format)"""
    x = distance * math.cos(azimuth) * math.cos(elevation)
    y = distance * math.sin(azimuth) * math.cos(elevation)
    z = distance * math.sin(elevation)
    return (x, y, z)


def save_images(object_file: str) -> None:
    """
    FIXED: Saves RGB, Depth, and Mask images in MVD-Fusion format
    """
    os.makedirs(args.output_dir, exist_ok=True)
    reset_scene()
    
    # Load object
    load_object(object_file)
    object_uid = os.path.basename(object_file).split(".")[0]
    normalize_scene()
    add_lighting()
    cam, cam_constraint = setup_camera()
    
    # Create empty object to track
    empty = bpy.data.objects.new("Empty", None)
    scene.collection.objects.link(empty)
    cam_constraint.target = empty

    # FIXED: Create nested views directory
    views_dir = os.path.join(args.output_dir, object_uid, "views")
    os.makedirs(views_dir, exist_ok=True)
    
    # NEW: Setup compositor for depth and mask
    depth_output, mask_output = setup_compositor_for_depth_and_mask(views_dir)

    # Render up to 64 views using MVD-Fusion camera poses
    num_views = min(args.num_images, 16)
    
    for i in range(num_views):
        # Use MVD-Fusion camera positions
        azimuth = AZIMUTHS_16[i]
        elevation = ELEVATIONS_16[i]
        
        # Set camera position
        cam_pos = get_camera_position(azimuth, elevation, args.camera_dist)
        cam.location = cam_pos
        
        # Set output paths
        rgb_path = os.path.join(views_dir, f"{i:03d}_rgb.jpg")  # FIXED: .jpg
        
        # Configure file outputs for this frame
        depth_output.file_slots[0].path = f"{i:03d}_depth"
        mask_output.file_slots[0].path = f"{i:03d}_mask"
        
        # Set RGB output
        scene.render.filepath = rgb_path
        
        # Render once - compositor handles all outputs
        bpy.ops.render.render(write_still=True)
        
        # Rename depth and mask files (Blender adds frame numbers)
        depth_temp = os.path.join(views_dir, f"{i:03d}_depth0001.png")
        depth_final = os.path.join(views_dir, f"{i:03d}_depth.png")
        mask_temp = os.path.join(views_dir, f"{i:03d}_mask0001.jpg")
        mask_final = os.path.join(views_dir, f"{i:03d}_mask.jpg")
        
        if os.path.exists(depth_temp):
            os.rename(depth_temp, depth_final)
        if os.path.exists(mask_temp):
            os.rename(mask_temp, mask_final)
        
        print(f"Rendered view {i:03d}/{num_views}")
    save_camera_parameters_mvd(views_dir, num_views, args.camera_dist, cam)
    print(f"✅ Completed {object_uid}: {num_views} views (RGB + Depth + Mask)")


def save_camera_parameters_mvd(views_dir, num_views, camera_dist=1.5, cam=None):
    """Save MVD-Fusion camera parameters with proper intrinsics"""
    camera_params = []
    
    # Calculate proper camera intrinsics
    focal_length = cam.data.lens
    sensor_width = cam.data.sensor_width
    sensor_height = sensor_width * (render.resolution_y / render.resolution_x)
    
    # Focal length in pixels
    f_x = (focal_length * render.resolution_x) / sensor_width
    f_y = (focal_length * render.resolution_y) / sensor_height
    
    # Principal point (center of image)
    c_x = render.resolution_x / 2
    c_y = render.resolution_y / 2
    
    for i in range(num_views):
        azimuth = AZIMUTHS_16[i]
        elevation = ELEVATIONS_16[i]
        
        # Set camera to the same position as during rendering
        cam_pos = get_camera_position(azimuth, elevation, camera_dist)
        cam.location = cam_pos
        
        # ✅ Get ACTUAL camera orientation after constraints
        bpy.context.view_layer.update()  # Ensure constraints are applied
        cam_matrix = cam.matrix_world
        
        # Extract actual look direction from camera matrix
        look_dir = cam_matrix @ Vector((0, 0, -1)) - cam_matrix.translation
        look_target = cam_matrix.translation + look_dir
        
        # Extract actual up vector from camera matrix  
        up_dir = cam_matrix @ Vector((0, 1, 0)) - cam_matrix.translation
        
        camera_params.append({
            'view_id': i,
            'azimuth': float(azimuth),
            'elevation': float(elevation),
            'camera_distance': float(camera_dist),
            'position': [float(cam_pos[0]), float(cam_pos[1]), float(cam_pos[2])],
            'target': [float(look_target.x), float(look_target.y), float(look_target.z)],
            'up': [float(up_dir.x), float(up_dir.y), float(up_dir.z)],
            'focal_length': [float(f_x), float(f_y)],
            'principal_point': [float(c_x), float(c_y)],
            'image_size': [render.resolution_x, render.resolution_y],
            'sensor_size': [sensor_width, sensor_height]
        })
    
    camera_file = os.path.join(views_dir, "cameras.json")
    with open(camera_file, 'w') as f:
        json.dump(camera_params, f, indent=2)
    print(f"✅ Saved camera parameters for {num_views} views")

def download_object(object_url: str) -> str:
    """Download the object and return the path."""
    uid = object_url.split("/")[-1].split(".")[0]
    tmp_local_path = os.path.join("tmp-objects", f"{uid}.glb.tmp")
    local_path = os.path.join("tmp-objects", f"{uid}.glb")
    os.makedirs(os.path.dirname(tmp_local_path), exist_ok=True)
    urllib.request.urlretrieve(object_url, tmp_local_path)
    os.rename(tmp_local_path, local_path)
    local_path = os.path.abspath(local_path)
    return local_path


if __name__ == "__main__":
    try:
        start_i = time.time()
        if args.object_path.startswith("http"):
            local_path = download_object(args.object_path)
        else:
            local_path = args.object_path
        save_images(local_path)
        end_i = time.time()
        print(f"✅ Finished {local_path} in {end_i - start_i:.1f} seconds")
        if args.object_path.startswith("http"):
            os.remove(local_path)
    except Exception as e:
        print(f"❌ Failed to render {args.object_path}")
        print(e)
        import traceback
        traceback.print_exc()