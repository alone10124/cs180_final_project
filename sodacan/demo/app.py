"""Gradio demo: film the can, watch the room appear.

  python -m sodacan.demo.app [--share] [--model runs/restore/unet/best.pt]

Upload (or record) a 10-second video of the can on its ChArUco board. The panorama
builds up frame by frame, then the enhanced version appears. Three panels: measured,
enhanced (labelled as prior-enhanced: the network can fill in plausible detail, not
measure new detail), and the reference room panorama if one is given. The 360 viewer
shows the measured panorama. The "Simulated capture" button renders a synthetic can
video with a known room, which doubles as the rehearsal and backup video.
"""
import argparse
import base64
import glob
import tempfile
from pathlib import Path

import cv2
import numpy as np
import torch

from sodacan import debug
from sodacan.color import auto_exposure, to_uint8
from sodacan.config import REPO_ROOT, config_dir, load_can, load_pipeline
from sodacan.demo.run import process
from sodacan.geometry.equirect import to_viewer

VIEWER = """<!doctype html><html><head><meta charset="utf-8">
<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/pannellum/2.5.6/pannellum.css">
<script src="https://cdnjs.cloudflare.com/ajax/libs/pannellum/2.5.6/pannellum.js"></script>
<style>html,body,#p{margin:0;width:100%;height:100%;background:#111}</style></head>
<body><div id="p"></div><script>
pannellum.viewer('p', {type: 'equirectangular', panorama: '__SRC__', autoLoad: true, hfov: 100, showZoomCtrl: true});
</script></body></html>"""


def viewer_html(img_rgb):
    ok, buf = cv2.imencode(".jpg", cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 92])
    src = "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode()
    doc = VIEWER.replace("__SRC__", src).replace("&", "&amp;").replace('"', "&quot;")
    return f'<iframe style="width:100%;height:420px;border:0;border-radius:8px" srcdoc="{doc}"></iframe>'


def choices(pattern):
    return sorted(str(Path(p)) for p in glob.glob(pattern))


def simulate_video(seed, can, path):
    """Render a synthetic capture (room with a volunteer, can on the board) to an mp4."""
    from sodacan.capture.board import load_board, render_texture
    from sodacan.capture.video import write_video
    from sodacan.geometry.camera import Intrinsics
    from sodacan.sim.dataset import random_label
    from sodacan.sim.fast_render import RenderParams, render
    from sodacan.sim.rooms import procedural_room
    from sodacan.sim.trajectories import arc_with_tilt

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    rng = np.random.default_rng(seed)
    env = procedural_room(1024, 2048, seed=seed, device=dev, person=True)
    K = Intrinsics(1280, 720, 1000.0, 1000.0, 639.5, 359.5)
    c2w = arc_with_tilt(100, distance=0.30, arc_deg=50, elevation_deg=(-2, 25), cycles=2, target_z=can.height / 2,
                        start_deg=float(rng.uniform(0, 360)), jitter_m=0.002, seed=seed, device=dev, min_z=0.015)
    _, bcfg = load_board()
    tex = render_texture(bcfg, ppm=2500)
    tex["texture"] = tex["texture"].to(dev)
    params = RenderParams(table=tex, roughness_deg=0.5, strength=1.0, label=random_label(rng, *can.atlas, dev),
                          ambient=0.6, exposure=0.7, noise_read=0.002, noise_shot=0.001, quantize=True, chunk=4)
    rgb = []
    for i in range(0, len(c2w), 10):  # chunked: a 4 GB GPU cannot hold every frame at once
        rgb += [to_uint8(f) for f in render(env, c2w[i : i + 10], K, can, params)]
    write_video(path, rgb, fps=10)
    phone = {"width": K.width, "height": K.height, "fx": K.fx, "fy": K.fy, "cx": K.cx, "cy": K.cy, "dist": [0] * 5}
    return env, phone


def build(default_model=None):
    import gradio as gr

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    models = {}

    def get_model(path):
        if not path or path == "none":
            return None
        if path not in models:
            from sodacan.restore.infer import load_model

            models[path] = load_model(path, device)
        return models[path]

    def run(video, phone, model_path, reference, sim_state):
        can = load_can()
        cfg = load_pipeline()
        H, W = cfg["pano"]
        ref = None
        phone_cfg = phone
        simulated = bool(sim_state)  # cleared when a real video is uploaded
        if simulated:
            phone_cfg = sim_state["phone"]
            ref = sim_state["env"]
        elif reference:
            from sodacan.capture.load import load_reference

            ref = load_reference(reference, H, W)
        if not video:
            yield "Upload a video or press 'Simulated capture' first.", None, None, None, None, "", ""
            return
        model = get_model(model_path)
        exposure = None
        for up in process(video, phone_cfg, "board", can, cfg, model, device):
            if up.pano is None:
                yield up.message, None, None, None, None, "", ""
                continue
            if exposure is None or up.stage == "done":
                exposure = auto_exposure(up.pano, up.valid)
            measured = debug.display(up.pano, up.valid, exposure)
            coverage = debug.heat(up.weight, up.valid)
            enhanced = debug.display(up.enhanced, up.valid, exposure) if up.enhanced is not None else None
            refimg = None
            if ref is not None and up.stage == "done":
                from sodacan.geometry.equirect import resize_pano
                from sodacan.metrics import align_azimuth

                r = resize_pano(ref.to(up.pano.device), H, W)
                if not simulated:  # a phone panorama has an unknown heading
                    r, _, _ = align_azimuth(up.pano, r, up.valid)
                refimg = to_uint8(r, auto_exposure(r, up.valid))
            view = viewer_html(np.ascontiguousarray(np.flip(measured, 1))) if up.stage == "done" else ""
            timing = ", ".join(f"{k} {v:.1f} s" for k, v in up.timing.items())
            yield f"{up.message}  [{timing}]", measured, coverage, enhanced, refimg, up.hint, view

    def make_sim(seed):
        can = load_can()
        path = str(Path(tempfile.gettempdir()) / f"sodacan_sim_{int(seed)}.mp4")
        env, phone = simulate_video(int(seed), can, path)
        return path, {"phone": phone, "env": env.cpu()}

    with gr.Blocks(title="Seeing the world in a soda can") as demo:
        gr.Markdown("## Seeing the world in a soda can\nA soda can is a 360 degree camera, if you know its shape. "
                    "Film the can on its board for about ten seconds, sweeping the phone sideways and tilting it up and down.")
        sim_state = gr.State({})
        with gr.Row():
            with gr.Column(scale=1):
                video = gr.Video(label="Can video", sources=["upload", "webcam"])
                phone = gr.Dropdown(choices(str(config_dir() / "phone_*.yaml")), value=str(config_dir() / "phone_example.yaml"),
                                    label="Phone calibration", allow_custom_value=True)
                model = gr.Dropdown(["none"] + choices(str(REPO_ROOT / "runs" / "restore" / "*" / "best.pt")),
                                    value=default_model or "none", label="Restoration model", allow_custom_value=True)
                reference = gr.Image(label="Reference room panorama (optional)", type="filepath")
                with gr.Row():
                    seed = gr.Number(value=7, label="Sim seed", precision=0)
                    simbtn = gr.Button("Simulated capture")
                go = gr.Button("Reconstruct", variant="primary")
            with gr.Column(scale=2):
                status = gr.Textbox(label="Status", interactive=False)
                measured = gr.Image(label="Measured panorama (what the can reflected)")
                enhanced = gr.Image(label="Prior-enhanced (network output; may add plausible, unmeasured detail)")
                ref_img = gr.Image(label="Reference")
                coverage = gr.Image(label="Coverage (dark blue: never observed)")
                hint = gr.Textbox(label="Where to move the phone next", interactive=False)
                viewer = gr.HTML(label="360 viewer")
        simbtn.click(make_sim, [seed], [video, sim_state])
        video.upload(lambda: {}, None, sim_state)
        video.clear(lambda: {}, None, sim_state)
        go.click(run, [video, phone, model, reference, sim_state], [status, measured, coverage, enhanced, ref_img, hint, viewer])
    return demo


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--share", action="store_true")
    ap.add_argument("--model")
    ap.add_argument("--port", type=int, default=7860)
    args = ap.parse_args()
    build(args.model).queue().launch(share=args.share, server_port=args.port, show_error=True)


if __name__ == "__main__":
    main()
