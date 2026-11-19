"""Blender Cycles render of a scene exported by sodacan.sim.blender_export (photoreal validation).

  blender -b -P sodacan/sim/blender_scene.py -- data/blender/scene_000 [--samples 512 --denoise]

Runs inside Blender's Python (bpy, numpy; no torch). Writes <scene>/blender/frame_XXXX.hdr.

The environment is mapped with math nodes to match geometry/equirect.py exactly
(column from atan2(y, x), row from the polar angle, z up). The can is a smooth
256-sided cylinder. Bare metal: Glossy BSDF (GGX). Printed can: Principled BSDF with
the label as base colour, mapped with the same surface atlas convention
(geometry/cylinder.py), dielectric specular with IOR 1.5. The wall is an open tube (no caps
sharing its vertices, which would bend its smooth normals) with a flat grey lid on top. Roughness in degrees is
converted to GGX roughness with r = sqrt(tan(sigma)), an approximation: the two lobes
are not the same shape. The table is a plain diffuse plane.
"""
import argparse
import json
import math
import sys
from pathlib import Path

import bpy
import numpy as np
from mathutils import Matrix


def math_node(tree, operation, *values):
    node = tree.nodes.new("ShaderNodeMath")
    node.operation = operation
    for socket, value in zip(node.inputs, values):
        if isinstance(value, (int, float)):
            socket.default_value = value
        else:
            tree.links.new(value, socket)
    return node.outputs[0]


def build_world(scene, env_path, strength):
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
    u = math_node(tree, "MULTIPLY_ADD", math_node(tree, "ARCTAN2", xyz.outputs["Y"], xyz.outputs["X"]), 1 / (2 * math.pi), 0.5)
    v = math_node(tree, "MULTIPLY_ADD", math_node(tree, "ARCCOSINE", xyz.outputs["Z"]), -1 / math.pi, 1.0)
    uv = tree.nodes.new("ShaderNodeCombineXYZ")
    tree.links.new(u, uv.inputs["X"])
    tree.links.new(v, uv.inputs["Y"])
    image = tree.nodes.new("ShaderNodeTexImage")
    image.image = bpy.data.images.load(str(env_path))
    image.interpolation = "Linear"
    image.extension = "REPEAT"
    tree.links.new(uv.outputs[0], image.inputs["Vector"])
    background = tree.nodes.new("ShaderNodeBackground")
    background.inputs["Strength"].default_value = strength
    tree.links.new(image.outputs["Color"], background.inputs["Color"])
    output = tree.nodes.new("ShaderNodeOutputWorld")
    tree.links.new(background.outputs[0], output.inputs["Surface"])


def can_material(spec, folder):
    rough = math.sqrt(math.tan(math.radians(max(spec["roughness_deg"], 1e-3))))
    mat = bpy.data.materials.new("can")
    mat.use_nodes = True
    tree = mat.node_tree
    tree.nodes.clear()
    out = tree.nodes.new("ShaderNodeOutputMaterial")
    if spec["material"]["model"] in ("metal", "mirror"):
        try:
            shader = tree.nodes.new("ShaderNodeBsdfAnisotropic")  # the Glossy BSDF since Blender 4.0
        except RuntimeError:
            shader = tree.nodes.new("ShaderNodeBsdfGlossy")
        f0 = spec["material"].get("f0", 1.0) * spec["strength"]
        shader.inputs["Color"].default_value = (f0, f0, f0, 1)
        shader.inputs["Roughness"].default_value = rough
    else:
        shader = tree.nodes.new("ShaderNodeBsdfPrincipled")
        shader.inputs["Roughness"].default_value = rough
        shader.inputs["IOR"].default_value = spec["material"].get("ior", 1.5)
        if spec.get("label"):
            # surface atlas: x = (atan2(y, x) + pi) / 2pi, y = z / height (Blender image v is up).
            # Object coordinates have their origin at the cylinder's centre, so z runs from -height/2.
            coord = tree.nodes.new("ShaderNodeTexCoord")
            xyz = tree.nodes.new("ShaderNodeSeparateXYZ")
            tree.links.new(coord.outputs["Object"], xyz.inputs[0])
            u = math_node(tree, "MULTIPLY_ADD", math_node(tree, "ARCTAN2", xyz.outputs["Y"], xyz.outputs["X"]), 1 / (2 * math.pi), 0.5)
            v = math_node(tree, "MULTIPLY_ADD", xyz.outputs["Z"], 1 / spec["can"]["height"], 0.5)
            uv = tree.nodes.new("ShaderNodeCombineXYZ")
            tree.links.new(u, uv.inputs["X"])
            tree.links.new(v, uv.inputs["Y"])
            image = tree.nodes.new("ShaderNodeTexImage")
            image.image = bpy.data.images.load(str(folder / spec["label"]))
            image.extension = "REPEAT"
            tree.links.new(uv.outputs[0], image.inputs["Vector"])
            tree.links.new(image.outputs["Color"], shader.inputs["Base Color"])
    tree.links.new(shader.outputs[0], out.inputs["Surface"])
    return mat


def build_camera(scene, spec):
    data = bpy.data.cameras.new("camera")
    W, H = spec["width"], spec["height"]
    data.sensor_fit = "HORIZONTAL"
    data.sensor_width = 36
    data.lens = spec["fx"] * 36 / W
    # OpenCV principal point (pixel centres at integers) -> Blender shift (fractions of the width)
    data.shift_x = ((W - 1) / 2 - spec["cx"]) / W
    data.shift_y = (spec["cy"] - (H - 1) / 2) / W
    data.clip_start = 0.005
    camera = bpy.data.objects.new("camera", data)
    scene.collection.objects.link(camera)
    scene.camera = camera
    return camera


def use_gpu(scene):
    """Render on the GPU when Cycles finds one (OptiX, then CUDA, HIP, Metal, oneAPI)."""
    try:
        prefs = bpy.context.preferences.addons["cycles"].preferences
    except KeyError:
        return
    for kind in ("OPTIX", "CUDA", "HIP", "METAL", "ONEAPI"):
        try:
            prefs.compute_device_type = kind
            prefs.get_devices()
        except (TypeError, ValueError):
            continue
        gpus = [d for d in prefs.devices if d.type == kind]
        if gpus:
            for d in prefs.devices:
                d.use = d.type == kind
            scene.cycles.device = "GPU"
            print(f"Cycles on {kind}: {', '.join(d.name for d in gpus)}")
            return


def main():
    argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else sys.argv[1:]
    ap = argparse.ArgumentParser()
    ap.add_argument("scene")
    ap.add_argument("--samples", type=int, default=128)
    ap.add_argument("--denoise", action="store_true",
                    help="OpenImageDenoise. A clear coat reflects ~4 %%, so label removal divides path-tracing noise by ~0.04")
    ap.add_argument("--out", default="blender", help="subfolder of the scene for the frames")
    args = ap.parse_args(argv)
    folder = Path(args.scene).resolve()
    spec = json.loads((folder / "scene.json").read_text())

    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene
    scene.render.engine = "CYCLES"
    scene.cycles.samples = args.samples
    use_gpu(scene)
    scene.cycles.use_denoising = args.denoise
    if args.denoise:
        scene.cycles.denoiser = "OPENIMAGEDENOISE"
    scene.render.resolution_x, scene.render.resolution_y = spec["width"], spec["height"]
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "HDR"
    scene.view_settings.view_transform = "Standard"
    build_world(scene, folder / "env.hdr", spec["exposure"])

    can = spec["can"]
    # The wall has no caps: with caps sharing its vertices, smooth shading would average the
    # caps' normals into the wall's and tilt it like a barrel (a window then reflects as a dot
    # instead of a vertical streak). The lid is a separate flat disk.
    bpy.ops.mesh.primitive_cylinder_add(vertices=256, radius=can["radius"], depth=can["height"], end_fill_type="NOTHING",
                                        location=(0, 0, can["height"] / 2))
    obj = bpy.context.object
    for poly in obj.data.polygons:
        poly.use_smooth = True
    obj.data.materials.append(can_material(spec, folder))
    bpy.ops.mesh.primitive_circle_add(vertices=256, radius=can["radius"], fill_type="NGON", location=(0, 0, can["height"]))
    lid = bpy.data.materials.new("lid")
    lid.use_nodes = True
    lid.node_tree.nodes["Principled BSDF"].inputs["Base Color"].default_value = (0.5, 0.5, 0.5, 1)
    bpy.context.object.data.materials.append(lid)
    bpy.ops.mesh.primitive_plane_add(size=4, location=(0, 0, 0))
    table = bpy.data.materials.new("table")
    table.use_nodes = True
    table.node_tree.nodes["Principled BSDF"].inputs["Base Color"].default_value = (0.35, 0.33, 0.30, 1)
    bpy.context.object.data.materials.append(table)

    camera = build_camera(scene, spec)
    flip = np.diag([1.0, -1.0, -1.0])  # OpenCV camera axes -> Blender (y up, z back)
    (folder / args.out).mkdir(exist_ok=True)
    for i, pose in enumerate(spec["poses"]):
        m = np.array(pose)
        m[:3, :3] = m[:3, :3] @ flip
        camera.matrix_world = Matrix(m.tolist())
        scene.render.filepath = str(folder / args.out / f"frame_{i:04d}.hdr")
        bpy.ops.render.render(write_still=True)


if __name__ == "__main__":
    main()
