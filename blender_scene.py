"""Render the photos of a synthetic scene in Blender (Cycles).

  python synth.py --name two --layout two                 # scene description and analytic photos
  blender -b -P blender_scene.py -- two                   # renders scenes/two/data/blender/frame_0001.hdr ...
  python synth.py --name two_blender --layout two --photos scenes/two/data/blender

The second synth.py call must repeat the first one's scene options; it rebuilds the
geometry buffers and takes the photos from Blender. Also runs as `python blender_scene.py two`
where the bpy module is installed.

The environment is mapped with explicit math nodes so that it matches SRMModel.lookup
exactly: column from atan2(y, x), row from the polar angle, z up. The cans are Glossy
BSDFs of constant reflectance; printed labels (--label) are not reproduced here, and
the table is shaded by the environment, unlike the flat table of the analytic photos.
"""
import argparse
import json
import math
import sys
from pathlib import Path

import bpy
import numpy as np
from mathutils import Matrix

ROOT = Path(__file__).resolve().parent


def math_node(tree, operation, *values):
    node = tree.nodes.new("ShaderNodeMath")
    node.operation = operation
    for socket, value in zip(node.inputs, values):
        if isinstance(value, (int, float)):
            socket.default_value = value
        else:
            tree.links.new(value, socket)
    return node.outputs[0]


def build_world(scene, env_path):
    world = bpy.data.worlds.new("environment")
    scene.world = world
    world.use_nodes = True
    tree = world.node_tree
    tree.nodes.clear()
    coord = tree.nodes.new("ShaderNodeTexCoord")
    unit = tree.nodes.new("ShaderNodeVectorMath")
    unit.operation = "NORMALIZE"
    tree.links.new(coord.outputs["Generated"], unit.inputs[0])
    xyz = tree.nodes.new("ShaderNodeSeparateXYZ")
    tree.links.new(unit.outputs["Vector"], xyz.inputs[0])
    azimuth = math_node(tree, "ARCTAN2", xyz.outputs["Y"], xyz.outputs["X"])
    u = math_node(tree, "MULTIPLY_ADD", azimuth, 1 / (2 * math.pi), 0.5)
    polar = math_node(tree, "ARCCOSINE", xyz.outputs["Z"])
    v = math_node(tree, "MULTIPLY_ADD", polar, -1 / math.pi, 1.0)  # image row 0 (zenith) is v = 1
    uv = tree.nodes.new("ShaderNodeCombineXYZ")
    tree.links.new(u, uv.inputs["X"])
    tree.links.new(v, uv.inputs["Y"])
    image = tree.nodes.new("ShaderNodeTexImage")
    image.image = bpy.data.images.load(str(env_path))
    image.interpolation = "Linear"
    image.extension = "REPEAT"
    tree.links.new(uv.outputs[0], image.inputs["Vector"])
    background = tree.nodes.new("ShaderNodeBackground")
    tree.links.new(image.outputs["Color"], background.inputs["Color"])
    output = tree.nodes.new("ShaderNodeOutputWorld")
    tree.links.new(background.outputs[0], output.inputs["Surface"])


def material(name, kind, colour, roughness=0.0):
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    tree = mat.node_tree
    tree.nodes.clear()
    if kind == "glossy":
        try:
            shader = tree.nodes.new("ShaderNodeBsdfAnisotropic")  # the Glossy BSDF since Blender 4.0
        except RuntimeError:
            shader = tree.nodes.new("ShaderNodeBsdfGlossy")
        shader.inputs["Roughness"].default_value = roughness
    else:
        shader = tree.nodes.new("ShaderNodeBsdfDiffuse")
    shader.inputs["Color"].default_value = (*colour, 1)
    output = tree.nodes.new("ShaderNodeOutputMaterial")
    tree.links.new(shader.outputs[0], output.inputs["Surface"])
    return mat


def smooth_walls(obj):
    for polygon in obj.data.polygons:
        polygon.use_smooth = len(polygon.vertices) == 4


def build_objects(spec, roughness):
    rho = spec["reflectance"]
    metal = material("can", "glossy", (rho, rho, rho), roughness)
    size = 2 * spec["floor_half"]
    bpy.ops.mesh.primitive_plane_add(size=size, location=(0, 0, 0))
    bpy.context.object.data.materials.append(material("table", "diffuse", (0.25, 0.23, 0.21)))
    for can in spec["cans"]:
        neck = can.get("neck", 0)
        body = can["height"] - neck
        bpy.ops.mesh.primitive_cylinder_add(vertices=256, radius=can["radius"], depth=body, location=(can["x"], can["y"], body / 2))
        smooth_walls(bpy.context.object)
        bpy.context.object.data.materials.append(metal)
        if neck > 0:
            bpy.ops.mesh.primitive_cone_add(
                vertices=256, radius1=can["radius"], radius2=can["top_radius"], depth=neck, location=(can["x"], can["y"], body + neck / 2)
            )
            smooth_walls(bpy.context.object)
            bpy.context.object.data.materials.append(metal)


def build_camera(scene, spec):
    data = bpy.data.cameras.new("camera")
    data.sensor_fit = "HORIZONTAL"
    data.sensor_width = 36
    data.lens = spec["focal"] * 36 / spec["width"]
    data.shift_x = (spec["width"] / 2 - spec["cx"]) / spec["width"]
    data.shift_y = (spec["cy"] - spec["height"] / 2) / spec["width"]
    data.clip_start = 0.01
    camera = bpy.data.objects.new("camera", data)
    scene.collection.objects.link(camera)
    scene.camera = camera
    return camera


def main():
    argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else sys.argv[1:]
    ap = argparse.ArgumentParser()
    ap.add_argument("name", help="scene under scenes/, created by synth.py")
    ap.add_argument("--samples", type=int, default=64)
    ap.add_argument("--frames", type=int, default=0, help="render only the first N frames")
    ap.add_argument("--roughness", type=float, default=0.0)
    args = ap.parse_args(argv)

    data = ROOT / "scenes" / args.name / "data"
    spec = json.loads((data / "scene.json").read_text())
    if spec.get("label"):
        print("warning: printed labels are not reproduced; cans render as bare metal")
    rows = np.loadtxt(data / "xy_poses.txt", skiprows=1, ndmin=2)

    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene
    scene.render.engine = "CYCLES"
    scene.cycles.samples = args.samples
    scene.cycles.use_denoising = False
    scene.render.resolution_x, scene.render.resolution_y = spec["width"], spec["height"]
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "HDR"
    scene.view_settings.view_transform = "Standard"
    build_world(scene, data / spec["env"])
    build_objects(spec, args.roughness)
    camera = build_camera(scene, spec)

    (data / "blender").mkdir(exist_ok=True)
    flip = np.diag([1.0, -1.0, -1.0])  # camera axes: ours x right, y down, z forward; Blender x right, y up, z back
    for row in rows[: args.frames or len(rows)]:
        matrix = np.eye(4)
        matrix[:3, :3] = row[4:].reshape(3, 3) @ flip
        matrix[:3, 3] = row[1:4]
        camera.matrix_world = Matrix(matrix.tolist())
        scene.render.filepath = str(data / "blender" / f"frame_{int(row[0]):04d}.hdr")
        bpy.ops.render.render(write_still=True)


if __name__ == "__main__":
    main()
