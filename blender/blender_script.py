"""Blender script to render images of 3D models.

This script is used to render images of 3D models. It takes in a list of paths
to .glb files and renders images of each model. The images are from rotating the
object around the origin. The images are saved to the output directory.

"""
# import bpy
import argparse
import json
import math
import os
import random
import sys
sys.path.append('..')
import time
import urllib.request
import uuid
from typing import Tuple
from mathutils import Vector, Matrix
import numpy as np
import bpy
from mathutils import Vector
from glob import glob
import traceback
import objaverse
import shutil
import torch
import cv2
import copy
from utils.pose_sample import RandomIterator, EulerAngleIterator
from scripts.generate_bg_and_rotate_envir_map import per_env_per_pose, read_hdr

# ENV_PATH = os.path.abspath('../../dataset/lighting/test_4')
# ENV_NAME = [
#     # 'none',
#     '012_hdrmaps_com_free_2K.exr', 
#     '045_hdrmaps_com_free_2K.exr', 
#     '087_hdrmaps_com_free_2K.exr', 
#     '109_hdrmaps_com_free_2K.exr'
# ]
# BASE_ENV = 0

DEPTH_SCALE = 1

context = bpy.context
scene = context.scene
render = scene.render

def get_environment(path):
    # Use '**/*.exr' to match .exr files in the specified path and all its subdirectories
    exr_list = glob(os.path.join(path, '**', '*.exr'), recursive=True)
    hdr_list = glob(os.path.join(path, '**', '*.hdr'), recursive=True)
    env_list = exr_list + hdr_list
    env_list.sort()
    env_list = [os.path.abspath(env) for env in env_list]
    return env_list

# add environment map as the lighting condition
def add_light_env(env=(1, 1, 1, 1), strength=1, rot_vec_rad=(0, 0, 0), scale=(1, 1, 1)):
    r"""Adds environment lighting.
    Args:
        env (tuple(float) or str, optional): Environment map. If tuple,
            it's RGB or RGBA, each element of which :math:`\in [0,1]`.
            Otherwise, it's the path to an image.
        strength (float, optional): Light intensity.
        rot_vec_rad (tuple(float), optional): Rotations in radians around x, y and z.
        scale (tuple(float), optional): If all changed simultaneously, then no effects.
    """
    engine = bpy.context.scene.render.engine
    assert engine == "CYCLES", "Rendering engine is not Cycles"

    if isinstance(env, str):
        bpy.data.images.load(env, check_existing=True)
        env = bpy.data.images[os.path.basename(env)]
    else:
        if len(env) == 3:
            env += (1,)
        assert len(env) == 4, "If tuple, env must be of length 3 or 4"

    world = bpy.context.scene.world
    world.use_nodes = True
    node_tree = world.node_tree
    nodes = node_tree.nodes
    links = node_tree.links

    bg_node = nodes.new("ShaderNodeBackground")
    links.new(bg_node.outputs["Background"], nodes["World Output"].inputs["Surface"])

    if isinstance(env, tuple):
        # Color
        bg_node.inputs["Color"].default_value = env
        print(("Environment is pure color, " "so rotation and scale have no effect"))
    else:
        # Environment map
        texcoord_node = nodes.new("ShaderNodeTexCoord")
        env_node = nodes.new("ShaderNodeTexEnvironment")
        env_node.image = env
        mapping_node = nodes.new("ShaderNodeMapping")
        mapping_node.inputs["Rotation"].default_value = rot_vec_rad
        mapping_node.inputs["Scale"].default_value = scale
        links.new(texcoord_node.outputs["Generated"], mapping_node.inputs["Vector"])
        links.new(mapping_node.outputs["Vector"], env_node.inputs["Vector"])
        links.new(env_node.outputs["Color"], bg_node.inputs["Color"])

    bg_node.inputs["Strength"].default_value = strength

    return env

def bpy_image_2_torch(env, size):
    # Get the pixel data as a flat array
    pixels = np.array(env.pixels[:])  # Get the pixel data
    width = env.size[0]
    height = env.size[1]

    image_array = pixels.reshape((height, width, 4))[:,:,3]

    # Convert to RGB by taking the first three channels
    rgb_resized = cv2.resize(image_array, size, interpolation=cv2.INTER_LINEAR)
    # Resize the image using Pillow

    # Convert the resized image back to a NumPy array
    rgb_tensor = torch.from_numpy(rgb_resized).to(torch.float32) 
    return rgb_tensor

def remove_unwanted_objects():
    """
    Remove unwanted objects from the scene, such as lights and background plane objects.
    """
    # Remove undesired objects and existing lights
    objs = []
    for o in bpy.data.objects:
        if o.name == 'BackgroundPlane':
            objs.append(o)
        elif o.type == 'LIGHT':
            objs.append(o)
        elif o.active_material is not None:
            for node in o.active_material.node_tree.nodes:
                if node.type == 'EMISSION':
                    objs.append(o)
               
    bpy.ops.object.delete({'selected_objects': objs})

def reset_scene():
    """Resets the scene to a clean state."""
    # delete everything that isn't part of a camera or a light
    for obj in bpy.data.objects:
        if obj.type not in {"CAMERA", "LIGHT"}:
            bpy.data.objects.remove(obj, do_unlink=True)
    # delete all the materials
    for material in bpy.data.materials:
        bpy.data.materials.remove(material, do_unlink=True)
    # delete all the textures
    for texture in bpy.data.textures:
        bpy.data.textures.remove(texture, do_unlink=True)
    # delete all the images
    for image in bpy.data.images:
        bpy.data.images.remove(image, do_unlink=True)

    scene.use_nodes = True

    nodes = bpy.context.scene.node_tree.nodes
    links = bpy.context.scene.node_tree.links
    # Clear default nodes
    for n in nodes:
        nodes.remove(n)

    # Create input render layer node
    render_layers = nodes.new("CompositorNodeRLayers")
    render_layers.label = 'Custom Outputs'
    render_layers.name = 'Custom Outputs'

    bpy.context.view_layer.use_pass_normal = True
    bpy.context.view_layer.use_pass_diffuse_color = True
    bpy.context.view_layer.use_pass_z = True

    depth_file_output = nodes.new(type="CompositorNodeOutputFile")
    map = nodes.new(type="CompositorNodeMapRange")
    depth_file_output.label = 'Depth Output'
    depth_file_output.name = 'Depth Output'
    depth_file_output.format.file_format = 'OPEN_EXR'
    depth_file_output.format.color_depth = '32'
    depth_file_output.format.exr_codec = 'ZIP'
    depth_file_output.base_path = "/"

    # Size is chosen kind of arbitrarily, try out until you're satisfied with resulting depth map.
    map.inputs['From Min'].default_value = 0
    map.inputs['From Max'].default_value = DEPTH_SCALE
    map.inputs['To Min'].default_value = 0
    map.inputs['To Max'].default_value = 1
    links.new(render_layers.outputs['Depth'], map.inputs[0])
    links.new(map.outputs[0], depth_file_output.inputs[0])

    # Create normal output nodes
    scale_node = nodes.new(type="CompositorNodeMixRGB")
    scale_node.blend_type = "MULTIPLY"
    # scale_node.use_alpha = True
    scale_node.inputs[2].default_value = (0.5, 0.5, 0.5, 1)
    links.new(render_layers.outputs["Normal"], scale_node.inputs[1])

    bias_node = nodes.new(type="CompositorNodeMixRGB")
    bias_node.blend_type = "ADD"
    # bias_node.use_alpha = True
    bias_node.inputs[2].default_value = (0.5, 0.5, 0.5, 0)
    links.new(scale_node.outputs[0], bias_node.inputs[1])

    alpha_normal = nodes.new(type="CompositorNodeSetAlpha")
    links.new(bias_node.outputs[0], alpha_normal.inputs["Image"])
    links.new(render_layers.outputs["Alpha"], alpha_normal.inputs["Alpha"])
    
    normal_file_output = nodes.new(type="CompositorNodeOutputFile")
    normal_file_output.label = "Normal Output"
    normal_file_output.base_path = "/"
    normal_file_output.file_slots[0].use_node_format = True
    normal_file_output.format.file_format = "PNG"    
    links.new(alpha_normal.outputs["Image"], normal_file_output.inputs[0])

    # Create albedo output nodes
    alpha_albedo = nodes.new(type="CompositorNodeSetAlpha")
    links.new(render_layers.outputs["DiffCol"], alpha_albedo.inputs["Image"])
    links.new(render_layers.outputs["Alpha"], alpha_albedo.inputs["Alpha"])

    albedo_file_output = nodes.new(type="CompositorNodeOutputFile")
    albedo_file_output.label = "Albedo Output"
    albedo_file_output.base_path = "/"
    # albedo_file_output.file_slots[0].use_node_format = True
    albedo_file_output.format.file_format = "PNG"
    albedo_file_output.format.color_mode = "RGB"
    albedo_file_output.format.color_depth = "8"
    links.new(alpha_albedo.outputs["Image"], albedo_file_output.inputs[0])
    
    # scene.view_settings.view_transform = 'Raw'
  
    return depth_file_output, normal_file_output, albedo_file_output

# load the glb model
def load_object(object_path: str) -> None:
    try:
        """Loads a glb model into the scene."""
        if object_path.endswith(".glb"):
            bpy.ops.import_scene.gltf(filepath=object_path, merge_vertices=True)
        elif object_path.endswith(".fbx"):
            bpy.ops.import_scene.fbx(filepath=object_path)
        else:
            raise ValueError(f"Unsupported file type: {object_path}")
        mesh_objects = [obj for obj in bpy.context.scene.objects if obj.type == 'MESH']
    except:
        os.system(f'echo "{object_path}" >> {args.output_dir}/bug.txt')
    return mesh_objects

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
        # Avoid that it scale CAMERA
        # In Neural Gaffer, they use decompose to let CAMERA World Pose to Rotation Euler and Location, then transfer to Matrix form
        if not obj.parent and obj.type not in {"CAMERA", "LIGHT"}:
            # print(obj)
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

    # Apply scale to matrix_world.
    bpy.context.view_layer.update()
    bbox_min, bbox_max = scene_bbox()
    offset = -(bbox_min + bbox_max) / 2
    for obj in scene_root_objects():
        obj.matrix_world.translation += offset
    bpy.ops.object.select_all(action="DESELECT")
    # return True

def pattern_file_exists(pattern: str) -> bool:
    """
    Check if any albedo file exists for index `j` in `save_path`.
    
    Matches files like: {save_path}/{j:03d}_albedo_*.png, .exr, .jpg, etc.
    
    Args:
        save_path (str): Directory path to search in.
        j (int): Index (will be formatted as 3-digit zero-padded number).
    
    Returns:
        bool: True if at least one matching file exists, False otherwise.
    """
    return bool(glob.glob(pattern))

def main(args):
    depth_file_output, normal_file_output, albedo_file_output = reset_scene()
    object_name = args.object_name
    # with open(args.objaverse_info, 'r') as file:
    #     object_info = json.load(file)[object_name]
    os.makedirs(os.path.join(args.output_dir, 'temp'),exist_ok=True)
    TEMP_PATH = os.path.join(args.output_dir, 'temp', f'temp_{args.object_name}.glb')
    # print(0)
    download(object_name, TEMP_PATH)
    # print(0)
    load_object(TEMP_PATH)
    remove_temp_file(TEMP_PATH)
    normalize_scene()

    camera = bpy.context.scene.camera
    camera.location = (0,1.5,0)
    
    cam_constraint = camera.constraints.new(type="TRACK_TO")
    cam_constraint.track_axis = "TRACK_NEGATIVE_Z"
    cam_constraint.up_axis = "UP_Y"

    b_empty = bpy.data.objects.new("Empty", None)
    b_empty.location = (0, 0, 0)
    camera.parent = b_empty

    bpy.context.scene.collection.objects.link(b_empty)
    bpy.context.view_layer.objects.active = b_empty
    cam_constraint.target = b_empty

    env_list = get_environment(args.lighting_dir)
    infos = {}
    infos['basic'] = \
        {
            "object_name": object_name,
            # "3D_model_path":object_info['save_path'],
            "focal": camera.data.lens,
            "sensor_size": [camera.data.sensor_width,camera.data.sensor_width] ,
            "image_size": [render.resolution_x, render.resolution_y],
            "environment": [env_list[i] for i in range(len(env_list))],
            "train_size": args.trainset_size,
            "test_size": args.testset_size,
            "depth_scale": DEPTH_SCALE
        }
    infos['train'] = []
    infos['test'] = []
    
    for i, env_path in enumerate(env_list):
        add_light_env(env_path)
        # resized_env_map = bpy_image_2_torch(env_map, size=(render.resolution_x, render.resolution_y))

        def render_image(euler_iter, split):
            assert split in ['train', 'test', 'valid'], False
            # for j, (location, direction) in enumerate(camera_iter):
            save_path = os.path.join(args.output_dir, 'objects', object_name, split)

            if not os.path.exists(save_path):
                os.makedirs(save_path)

            for j, euler in enumerate(euler_iter):
                b_empty.rotation_euler = euler
                bpy.context.view_layer.update()
                # In Neural Gaffer, they use decompose to let CAMERA World Pose to Rotation Euler and Location, then transfer to Matrix form
                # But in this work, we prevent scaling on Camera
                # pose: 001_pose.npy
                pose = np.array(camera.matrix_world)

                if args.pose:
                    pose_save_path = os.path.join(save_path, f'{j:03d}_pose')
                    np.save(os.path.join(pose_save_path), pose)

                infos[split].append(
                            {   
                                "image_name": f'{j:03d}_{i:03d}',
                                'pose_id': f"{j:03d}",
                                'environment': env_path,
                                "transform": copy.deepcopy(pose.tolist()),
                            }
                        )

                # image: 001_003_image.png
                if not args.no_rgb:
                    scene.render.filepath = os.path.join(save_path, f'{j:03d}_{i:03d}_image')
                else:
                    # 清空 filepath，避免寫出 RGB
                    scene.render.filepath = ''  
                    if i > 0: # not write depth or albedo anymore
                        continue


                if args.skip_exist and os.path.exists(scene.render.filepath+'.png'):
                    try:
                        img = cv2.imread(scene.render.filepath+'.png')
                        if img is not None:
                            height, width, _ = img.shape
                            if (width, height) == (render.resolution_x, render.resolution_y):
                                if i == 0:
                                    normal_exist = args.albedo and pattern_file_exists(os.path.join(save_path, f'{j:03d}_normal_*.png'))
                                    albedo_exist = args.albedo and pattern_file_exists(os.path.join(save_path, f'{j:03d}_albedo_*.png'))
                                    depth_exist = args.albedo and pattern_file_exists(os.path.join(save_path, f'{j:03d}_depth_*.png'))
                                    if normal_exist and albedo_exist and depth_exist:
                                        continue                               
                                else:
                                    continue
                            else:
                                raise ValueError(f"The image size is {width}x{height} pixels, \
                                                 not {render.resolution_x}x{render.resolution_x}")
                    except Exception as e:
                        print(e)

                # normal: 001_normal.png
                # albedo: 001_albedo.png
                # depth: 001_depth.exr
                if i == 0:
                    if args.normal:
                        normal_file_output.file_slots[0].use_node_format = True
                        normal_file_output.file_slots[0].path = \
                            os.path.join(save_path, f'{j:03d}_normal_')
                    if args.albedo:
                        albedo_file_output.file_slots[0].use_node_format = True
                        albedo_file_output.file_slots[0].path = \
                            os.path.join(save_path, f'{j:03d}_albedo_')
                    if args.depth:
                        depth_file_output.file_slots[0].use_node_format = True
                        depth_file_output.file_slots[0].path = \
                            os.path.join(save_path, f'{j:03d}_depth_')
                else:
                    bpy.context.view_layer.use_pass_normal = False
                    bpy.context.view_layer.use_pass_diffuse_color = False
                    bpy.context.view_layer.use_pass_z = False

                    depth_file_output.file_slots[0].use_node_format = False
                    depth_file_output.file_slots[0].path = ''

                    normal_file_output.file_slots[0].use_node_format = False
                    normal_file_output.file_slots[0].path = ''

                    albedo_file_output.file_slots[0].use_node_format = False
                    albedo_file_output.file_slots[0].path = ''

                    nodes = bpy.context.scene.node_tree.nodes
                    # Clear default nodes
                    # for n in nodes:
                    #     nodes.remove(n)


                # hdr: 001_003_hdr.png
                # ldr: 001_003_ldr.png
                if args.env:
                    target_envir_map_hdr, target_envir_map_ldr, target_envir_map_raw = \
                        per_env_per_pose(env_path, pose[:3,:3], c2w=True, \
                                        size=[render.resolution_x, render.resolution_y])
                    env_map_save_path = os.path.join(args.output_dir, 'environments', split)
                    if not os.path.exists(env_map_save_path):
                        os.makedirs(env_map_save_path)
                    target_envir_map_hdr.save(os.path.join(env_map_save_path, f'{j:03d}_{i:03d}_hdr.png'))
                    target_envir_map_ldr.save(os.path.join(env_map_save_path, f'{j:03d}_{i:03d}_ldr.png'))
                    np.save(os.path.join(env_map_save_path, f'{j:03d}_{i:03d}_raw'), target_envir_map_raw)
                
                bpy.ops.render.render(write_still=True)
        
        train_euler = RandomIterator(total_points=args.trainset_size, seed=666)
        render_image(train_euler, 'train')
        test_euler = RandomIterator(total_points=args.testset_size, seed=888)
        render_image(test_euler, 'test')

    json_output = json.dumps(infos, indent=4)
    os.makedirs(os.path.join(args.output_dir, 'objects', object_name), exist_ok=True)
    output_file_path = os.path.join(args.output_dir, 'objects', object_name, "info.json")  # 修改为你的文件路径
    with open(output_file_path, 'w') as json_file:
        json_file.write(json_output)
    print("Dataset info. have been save to", output_file_path)

def download(uid, tmp):
    objects = objaverse.load_objects(
        uids=[uid],
        download_processes=1
        )
    # annotations = objaverse.load_annotations(uid)
    
    file_path = objects[uid]
    if os.path.isfile(file_path): 
        # new_file_path = os.path.join(args.path, os.path.basename(file_path))
        shutil.copy(file_path, tmp)
        os.remove(file_path)
        
    # return {
    #     'uid': uid,
    #     'uri': annotations[uid]['uri'],
    #     'name':  annotations[uid]['name']
    # }

def remove_temp_file(tmp):
    if os.path.isfile(tmp): 
        os.remove(tmp)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--object_name",
        type=str,
        required=True,
        help="Object UID",
    )
    parser.add_argument(
        "--objaverse_info",
        type=str,
        default='../../dataset/Objaverse/objects.json',
        help="Path to the Objaverse info json",
    )
    # parsser.add_argument("--output_dir", type=str, default="{args.output_dir}/views_whole_sphere")
    parser.add_argument("--output_dir", type=str, default="../../dataset/relitObjaverse")
    parser.add_argument("--lighting_dir", type=str, default="../../dataset/lighting")


    parser.add_argument(
        "--engine", type=str, default="CYCLES", choices=["CYCLES", "BLENDER_EEVEE"]
    )
    parser.add_argument("--scale", type=float, default=0.8)
    parser.add_argument("--trainset_size", type=int, default=16)
    parser.add_argument("--testset_size", type=int, default=16)

    parser.add_argument("--image_size", type=tuple, default=(512, 512))


    parser.add_argument("--depth", action='store_true')
    parser.add_argument("--normal", action='store_true')
    parser.add_argument("--albedo", action='store_true')
    parser.add_argument("--pose", action='store_true')
    parser.add_argument("--no_rgb", action='store_true')


    parser.add_argument("--env", action='store_true')
    parser.add_argument("--skip_exist", action='store_true')

    argv = sys.argv[sys.argv.index("--") + 1 :]
    args = parser.parse_args(argv)

    print('===================', args.engine, '===================')

    # cam = scene.objects["Camera"]
    # cam.location = (0, 1.2, 0)
    # cam.data.lens = 35
    # cam.data.sensor_width = 32

    # cam_constraint = cam.constraints.new(type="TRACK_TO")
    # cam_constraint.track_axis = "TRACK_NEGATIVE_Z"
    # cam_constraint.up_axis = "UP_Y"

    # camera = bpy.context.scene.camera
    # camera.location = (0,1.5,0)

    # cam_constraint = camera.constraints.new(type="TRACK_TO")
    # cam_constraint.track_axis = "TRACK_NEGATIVE_Z"
    # cam_constraint.up_axis = "UP_Y"


    render.engine = args.engine
    render.image_settings.file_format = "PNG"
    render.image_settings.color_mode = "RGBA"
    render.resolution_x = 256
    render.resolution_y = 256
    render.resolution_percentage = 100

    scene.cycles.device = "GPU"
    scene.cycles.samples = 128
    scene.cycles.diffuse_bounces = 1
    scene.cycles.glossy_bounces = 1
    scene.cycles.transparent_max_bounces = 3
    scene.cycles.transmission_bounces = 3
    scene.cycles.filter_width = 0.01
    scene.cycles.use_denoising = True
    scene.render.film_transparent = True

    bpy.context.preferences.addons["cycles"].preferences.get_devices()
    # Set the device_type

    bpy.context.preferences.addons["cycles"].preferences.compute_device_type = "CUDA" # or "OPENCL"
    args.output_dir = os.path.abspath(args.output_dir)

    try:
        main(args)
    except Exception as E:
        print(E)
