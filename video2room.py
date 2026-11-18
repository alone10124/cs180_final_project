"""A video around a shiny object (a bag of chips) -> its environment panorama -> a 3D Gaussian room.

  conda activate chips
  python video2room.py --video bag.mp4 --name bag --quick    # smoke test, about an hour
  python video2room.py --video bag.mp4 --name bag            # full run
  python video2room.py --video bag.mp4 --name bag --view     # ... and open the viewer at the end
  python view_gaussians.py scenes/bag/results/room/3DGS.ply  # view a finished run

Stages, each a separate process with SCENE=<name>, outputs under scenes/<name>/:

  frames   video_frames.py       sharpest frame per window       capture/frames/
  sfm      sfm.py                COLMAP poses + dense points      capture/colmap/dense/
  mesh     video_scene.py mesh   aligned object + table mesh      data/
  buffers  video_scene.py buffers per-frame geometry, diffuse     cache/
  env      train.py srm + eval_env.py (or splat.py)               results/srm/   (part a)
  fill     inpaint.py            fill what the object never saw   results/srm_diffusion/
  room     Pano2Room             panorama -> 3D Gaussians         results/room/3DGS.ply  (part b)
  view     view_gaussians.py     browser viewer (with --view)

A stage whose output exists is skipped; --from STAGE reruns from there on, --to STAGE stops.
Logs: scenes/<name>/logs/<stage>.log. Setup: setup_linux.sh. Details: README.md.
"""
import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parent
STAGES = ["frames", "sfm", "mesh", "buffers", "env", "fill", "room", "view"]
PANO_SIZE = (2048, 1024)  # Pano2Room resizes its input to 2048 wide
P2R_OUTPUTS = ["3DGS.ply", "GS_render_video.mp4", "GS_depth_video.mp4"]


class Pipeline:
    def __init__(self, args):
        self.args = args
        self.base = ROOT / "scenes" / args.name
        self.frames = self.base / "capture" / "frames"
        self.colmap = self.base / "capture" / "colmap"
        self.dense = self.colmap / "dense"
        self.data, self.cache, self.results = self.base / "data", self.base / "cache", self.base / "results"
        self.logs = self.base / "logs"
        self.env_run = args.env_method  # results/srm or results/splat
        self.filled = self.results / f"{self.env_run}_{args.fill}"
        self.room = self.results / "room"
        self.env = dict(os.environ, SCENE=args.name, CUDA_VISIBLE_DEVICES=str(args.gpu))

    def done(self, stage):
        output = {
            "frames": self.frames / "frame_0001.jpg",
            "sfm": self.dense / "sparse_txt" / "images.txt",
            "mesh": self.data / "mesh.npz",
            "buffers": self.cache / "meta.json",
            "env": self.results / self.env_run / "valid.npy",
            "fill": self.filled / "filled.png",
            "room": self.room / "3DGS.ply",
        }.get(stage)
        return output is not None and output.exists()

    def run(self, stage, command, cwd=ROOT):
        """Run a command, echoing its output and appending it to logs/<stage>.log."""
        self.logs.mkdir(parents=True, exist_ok=True)
        log = self.logs / f"{stage}.log"
        command = [str(c) for c in command]
        print(f"[{stage}] {' '.join(command)}", flush=True)
        with open(log, "a") as sink:
            sink.write(f"\n$ {' '.join(command)}\n")
            proc = subprocess.Popen(command, cwd=cwd, env=self.env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
            for line in proc.stdout:
                sys.stdout.write(line)
                sink.write(line)
            code = proc.wait()
        if code:
            sys.exit(f"[{stage}] failed with exit code {code}; see {log}")

    def python(self, script, *args):
        return [sys.executable, ROOT / script, *args]

    # ------------------------------------------------------------ stages

    def stage_frames(self):
        assert self.args.video, "--video is required to extract frames"
        video = Path(self.args.video).expanduser().resolve()  # stages run from the repo root
        assert video.exists(), f"{video} not found"
        shutil.rmtree(self.frames, ignore_errors=True)
        self.run("frames", self.python("video_frames.py", video, self.frames, "--count", self.args.frames))

    def stage_sfm(self):
        self.run("sfm", self.python("sfm.py", self.frames, self.colmap, "--matcher", self.args.matcher, "--max-size", self.args.colmap_size))

    def stage_mesh(self):
        self.run("mesh", self.python("video_scene.py", "mesh", "--dense", self.dense, "--object-height", self.args.object_height))

    def stage_buffers(self):
        self.run("buffers", self.python("video_scene.py", "buffers"))

    def stage_env(self):
        if self.args.env_method == "splat":
            self.run("env", self.python("splat.py", "--cans", "a", "--out", "splat"))
            return
        train = ["--run", "srm", "--cans", "a", "--materials", self.args.materials, "--max-steps", self.args.srm_steps]
        train += ["--lr-srm", self.args.lr_srm, *self.args.train_args.split()]
        self.run("env", self.python("train.py", "srm", *train))
        self.run("env", self.python("eval_env.py", "--run", "srm"))

    def stage_fill(self):
        self.run("fill", self.python("inpaint.py", "--input", self.env_run, "--method", self.args.fill, "--out-width", PANO_SIZE[0]))

    def stage_room(self):
        p2r = Path(self.args.p2r_dir).resolve()
        assert (p2r / "pano2room.py").exists(), f"Pano2Room not found at {p2r}; run setup_linux.sh"
        self.room.mkdir(parents=True, exist_ok=True)

        # The repo's panoramas are mirrored relative to a camera panorama (see README, Conventions).
        image = cv2.imread(str(self.filled / "filled.png"))
        assert image is not None, f"cannot read {self.filled / 'filled.png'}"
        pano = cv2.resize(image[:, ::-1], PANO_SIZE, interpolation=cv2.INTER_CUBIC)
        cv2.imwrite(str(self.room / "input_panorama.png"), pano)

        # Pano2Room reads and writes fixed paths inside its own folder: one run at a time.
        target = p2r / "input" / "input_panorama.png"
        backup = p2r / "input" / "input_panorama.demo.png"
        if target.exists() and not backup.exists():
            shutil.copy(target, backup)
        shutil.copy(self.room / "input_panorama.png", target)
        # A fine-tuned inpainter left in output/SDFT_weights is loaded automatically; it belongs to another panorama.
        shutil.rmtree(p2r / "output" / "SDFT_weights", ignore_errors=True)
        shutil.rmtree(p2r / "output" / "Pano2Room-results", ignore_errors=True)

        python = self.p2r_python()
        if self.args.sdft:
            self.run("room", [*python, "create_SDFT_pairs.py"], cwd=p2r)
            self.run("room", [*python, "-m", "accelerate.commands.launch", "--config_file", "scripts/accelerate.yaml", "train_SDFT.py"], cwd=p2r)
        self.run("room", [*python, "pano2room.py"], cwd=p2r)
        for name in P2R_OUTPUTS:
            source = p2r / "output" / "Pano2Room-results" / name
            if source.exists():
                shutil.copy(source, self.room / name)
        assert (self.room / "3DGS.ply").exists(), f"Pano2Room wrote no 3DGS.ply; see {self.logs / 'room.log'}"

    def stage_view(self):
        command = self.python("view_gaussians.py", self.room / "3DGS.ply", "--port", self.args.port)
        print(f"[view] {' '.join(map(str, command))}", flush=True)
        subprocess.run([str(c) for c in command], env=self.env)

    def p2r_python(self):
        if self.args.p2r_python:
            return [self.args.p2r_python]
        conda = os.environ.get("CONDA_EXE") or shutil.which("conda")
        assert conda, "conda not found; pass --p2r-python /path/to/pano2room/env/bin/python"
        return [conda, "run", "--no-capture-output", "-n", self.args.p2r_env, "python"]

    # ------------------------------------------------------------ driver

    def __call__(self):
        args = self.args
        first = STAGES.index(args.from_stage) if args.from_stage else 0
        last = STAGES.index(args.to_stage) if args.to_stage else STAGES.index("view" if args.view else "room")
        for i in range(first, last + 1):
            stage = STAGES[i]
            rerun = args.force or (args.from_stage is not None)
            if stage != "view" and self.done(stage) and not rerun:
                print(f"[{stage}] done, skipping (use --from {stage} to rerun)", flush=True)
                continue
            getattr(self, f"stage_{stage}")()
        print(f"\nOutputs in {self.base}:")
        for label, path in [
            ("alignment check", self.data / "check_*.jpg"),
            ("recovered panorama", self.results / self.env_run / "pano.png"),
            ("filled panorama", self.filled / "filled.png"),
            ("3D Gaussians", self.room / "3DGS.ply"),
            ("fly-through video", self.room / "GS_render_video.mp4"),
        ]:
            print(f"  {label:20s} {path}")
        print(f"View: python view_gaussians.py {self.room / '3DGS.ply'}  (or drag it into https://superspl.at/editor)")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--video", help="MP4 (or any video OpenCV reads) orbiting the object")
    ap.add_argument("--name", required=True, help="scene name; outputs go to scenes/<name>/")
    ap.add_argument("--from", dest="from_stage", choices=STAGES, help="rerun from this stage on")
    ap.add_argument("--to", dest="to_stage", choices=STAGES, help="stop after this stage")
    ap.add_argument("--force", action="store_true", help="rerun stages even if their output exists")
    ap.add_argument("--quick", action="store_true", help="smoke test: 60 frames, 1000 SRM steps, telea fill")
    ap.add_argument("--gpu", default="0", help="GPU index for every stage")
    # part (a)
    ap.add_argument("--frames", type=int, default=300, help="frames kept from the video")
    ap.add_argument("--matcher", choices=["sequential", "exhaustive"], default="sequential")
    ap.add_argument("--colmap-size", type=int, default=1600, help="longest image side for COLMAP")
    ap.add_argument("--object-height", type=float, default=0.25, help="real height of the object in metres")
    ap.add_argument("--env-method", choices=["srm", "splat"], default="srm", help="srm: the paper (hours); splat: training-free preview (minutes)")
    ap.add_argument("--materials", type=int, default=2, help="SRM basis count (printed bag: ink and foil)")
    ap.add_argument("--srm-steps", type=int, default=30000)
    ap.add_argument("--lr-srm", type=float, default=5e-3)
    ap.add_argument("--train-args", default="", help="extra train.py arguments, e.g. '--lambda-g 0'")
    ap.add_argument("--fill", choices=["diffusion", "telea"], default="diffusion")
    # part (b)
    ap.add_argument("--p2r-dir", default=str(ROOT / "third_party" / "Pano2Room"))
    ap.add_argument("--p2r-env", default="pano2room", help="conda environment of Pano2Room")
    ap.add_argument("--p2r-python", default=None, help="python of the Pano2Room environment (instead of conda run)")
    ap.add_argument("--sdft", action="store_true", help="fine-tune Pano2Room's inpainter on this panorama first (slower, better)")
    ap.add_argument("--view", action="store_true", help="open the viewer at the end")
    ap.add_argument("--port", type=int, default=8080)
    args = ap.parse_args()
    if args.quick:
        args.frames, args.srm_steps, args.fill = min(args.frames, 60), min(args.srm_steps, 1000), "telea"
    Pipeline(args)()


if __name__ == "__main__":
    main()
