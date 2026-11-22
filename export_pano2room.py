"""Export a filled reflection panorama for the official Pano2Room implementation."""

import argparse
from pathlib import Path

from PIL import Image, ImageOps

from common import RESULTS, ROOT


def export_panorama(source, output, demo=False):
    with Image.open(source) as image:
        if demo:
            # The checked-in comparison stacks recovered, filled, and reference.
            panel_height = image.width // 2
            if image.width % 2 or image.height != 3 * panel_height:
                raise ValueError("expected three stacked 2:1 demo panoramas")
            image = image.crop((0, panel_height, image.width, 2 * panel_height))
        if image.width != 2 * image.height:
            raise ValueError("expected a single 2:1 equirectangular panorama")

        # Our azimuth increases left to right, opposite a camera panorama.
        camera_panorama = ImageOps.mirror(image.convert("RGB"))

    output.parent.mkdir(parents=True, exist_ok=True)
    camera_panorama.save(output, format="PNG")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--input", type=Path, help="filled panorama image or scene result folder")
    source.add_argument("--demo", action="store_true", help="use the saved synthetic Telea fill")
    parser.add_argument("--output", type=Path, default=RESULTS / "pano2room" / "input_panorama.png")
    args = parser.parse_args()

    if args.demo:
        path = ROOT / "previews" / "inpaint_telea_compare.png"
    else:
        path = args.input
        if not path.exists():
            path = RESULTS / path
        if path.is_dir():
            path = path / "filled.png"
    try:
        if path.resolve() == args.output.resolve():
            raise ValueError("use a different output path to preserve the source panorama")
        export_panorama(path, args.output, demo=args.demo)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    print(f"Wrote {args.output.resolve()}")


if __name__ == "__main__":
    main()
