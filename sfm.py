"""Camera poses and a dense point cloud from frames, with COLMAP (needs a CUDA build).

  python sfm.py scenes/bag/capture/frames scenes/bag/capture/colmap

Runs feature extraction, matching, mapping, undistortion, PatchMatch stereo and fusion.
Writes <workspace>/dense/images/ (undistorted frames), dense/sparse_txt/ (PINHOLE model
as text, read by capture.read_colmap) and dense/fused.ply (points with normals).
Option names differ between COLMAP versions, so each is looked up in the command's -h.
"""
import argparse
import os
import shutil
import struct
import subprocess
from functools import lru_cache
from pathlib import Path

ENV = dict(os.environ, QT_QPA_PLATFORM="offscreen")


@lru_cache(maxsize=None)
def help_text(command):
    result = subprocess.run(["colmap", command, "-h"], capture_output=True, text=True, env=ENV)
    return result.stdout + result.stderr


def option(command, *names):
    """The first of `names` that this COLMAP's `command` accepts, e.g. SiftExtraction.use_gpu."""
    for name in names:
        if f"--{name}" in help_text(command):
            return f"--{name}"
    raise RuntimeError(f"colmap {command} accepts none of {names}; check `colmap {command} -h`")


def colmap(command, args):
    line = ["colmap", command, *[str(a) for a in args]]
    print(" ".join(line), flush=True)
    subprocess.run(line, check=True, env=ENV)


def check_colmap():
    assert shutil.which("colmap"), "colmap not found; run setup_linux.sh and `conda activate chips`"
    result = subprocess.run(["colmap", "-h"], capture_output=True, text=True, env=ENV)
    banner = " ".join((result.stdout + result.stderr).split()[:40])
    if "without CUDA" in banner:
        raise RuntimeError(f"this COLMAP has no CUDA support, which dense stereo needs: {banner}")
    if "with CUDA" not in banner:
        print(f"warning: cannot tell whether COLMAP has CUDA support: {banner}", flush=True)


def registered(model):
    """Number of registered images in a binary COLMAP model (0 if unreadable)."""
    try:
        with open(Path(model) / "images.bin", "rb") as f:
            return struct.unpack("<Q", f.read(8))[0]
    except (OSError, struct.error):
        return 0


def run(frames, workspace, matcher="sequential", max_size=1600):
    """Returns (number of frames, number registered)."""
    check_colmap()
    frames, workspace = Path(frames), Path(workspace)
    workspace.mkdir(parents=True, exist_ok=True)
    database, sparse, dense = workspace / "database.db", workspace / "sparse", workspace / "dense"
    count = len(list(frames.glob("*.jpg")))

    if database.exists():
        database.unlink()
    colmap("feature_extractor", [
        "--database_path", database, "--image_path", frames,
        "--ImageReader.single_camera", 1, "--ImageReader.camera_model", "OPENCV",
        option("feature_extractor", "FeatureExtraction.use_gpu", "SiftExtraction.use_gpu"), 1,
        option("feature_extractor", "FeatureExtraction.max_image_size", "SiftExtraction.max_image_size"), max_size,
    ])
    gpu = option(f"{matcher}_matcher", "FeatureMatching.use_gpu", "SiftMatching.use_gpu")
    extra = ["--SequentialMatching.overlap", 20] if matcher == "sequential" else []
    colmap(f"{matcher}_matcher", ["--database_path", database, gpu, 1, *extra])

    shutil.rmtree(sparse, ignore_errors=True)
    sparse.mkdir()
    colmap("mapper", ["--database_path", database, "--image_path", frames, "--output_path", sparse])
    models = sorted(sparse.iterdir(), key=registered, reverse=True)
    assert models, "COLMAP could not register the frames; see the log"
    best = models[0]
    count_registered = registered(best)

    shutil.rmtree(dense, ignore_errors=True)
    colmap("image_undistorter", [
        "--image_path", frames, "--input_path", best, "--output_path", dense,
        "--output_type", "COLMAP", "--max_image_size", max_size,
    ])
    colmap("patch_match_stereo", [
        "--workspace_path", dense, "--workspace_format", "COLMAP",
        "--PatchMatchStereo.geom_consistency", "true", "--PatchMatchStereo.max_image_size", max_size,
    ])
    colmap("stereo_fusion", [
        "--workspace_path", dense, "--workspace_format", "COLMAP",
        "--input_type", "geometric", "--output_path", dense / "fused.ply",
    ])
    (dense / "sparse_txt").mkdir(exist_ok=True)
    colmap("model_converter", ["--input_path", dense / "sparse", "--output_path", dense / "sparse_txt", "--output_type", "TXT"])
    return count, count_registered


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("frames", help="directory of frame_0001.jpg ...")
    ap.add_argument("workspace", help="COLMAP workspace directory")
    ap.add_argument("--matcher", choices=["sequential", "exhaustive"], default="sequential")
    ap.add_argument("--max-size", type=int, default=1600, help="longest image side used by COLMAP")
    args = ap.parse_args()
    count, done = run(args.frames, args.workspace, args.matcher, args.max_size)
    print(f"registered {done}/{count} frames")
    if done < 0.8 * count:
        print("warning: fewer than 80% of frames registered; reshoot slower or with more texture on the table")


if __name__ == "__main__":
    main()
