"""Where the cans are in each photo: an open-vocabulary detector (OWLv2, "a soda can") for
boxes and SAM, prompted with those boxes, for the masks.

A chrome can has no appearance of its own (it shows the room), so neither SfM points nor
edges inside it can be trusted; its outline can. Mask-based fitting of a known model is how
recent work poses shiny objects (Accidental Light Probes, CVPR 2023: mask + photometric loss
on a known can model; multi-view mask alignment with a shared pose, e.g. Shugurov et al.
2022). Segmentation only: no restoration model is involved.

Models (Hugging Face, downloaded on first use): google/owlv2-base-patch16-ensemble and
facebook/sam-vit-base.
"""
import numpy as np
import torch

# a can seen end on (its lid or its concave bottom) is a disc: asked for by name
CAN_QUERIES = ["a soda can", "an aluminum drink can", "a metal can lying on its side",
               "the top of a soda can", "the bottom of an aluminum can"]
# things around the cans the detector would otherwise call a can: a box that matches one of
# these better than any can query is dropped
OTHER_QUERIES = ["a bottle", "a cup", "an instant noodle cup", "a paper bowl", "a food package",
                 "a tablet computer", "a clipboard", "a cardboard sheet", "a person", "a chair"]


def _post_process(proc, out, score, sizes):
    """Boxes, scores and query indices; the call moved between transformers versions."""
    ip = getattr(proc, "image_processor", None)
    for f in (getattr(ip, "post_process_object_detection", None), getattr(proc, "post_process_object_detection", None)):
        if f is not None:
            return f(outputs=out, threshold=score, target_sizes=sizes)[0]
    res = proc.post_process_grounded_object_detection(outputs=out, threshold=score, target_sizes=sizes)[0]
    if "labels" not in res or not torch.is_tensor(res["labels"]):
        res["labels"] = res.get("label_ids", res.get("labels"))
    return res


def detect_can_masks(images, device="cpu", long_side=768, score=0.12, max_cans=6, log=print):
    """images: list of HxWx3 uint8 RGB. Returns (masks, boxes, instances): per image a HxW
    bool union mask of the cans, an (N, 4) array of their boxes and an (N, H, W) bool array of
    each can's own mask, at the images' own size; or None if the models cannot be loaded."""
    try:
        import cv2
        from torchvision.ops import nms
        from transformers import Owlv2ForObjectDetection, Owlv2Processor, SamModel, SamProcessor
    except Exception as e:  # noqa: BLE001
        log(f"can masks: models unavailable ({e})")
        return None
    try:
        oproc = Owlv2Processor.from_pretrained("google/owlv2-base-patch16-ensemble")
        owl = Owlv2ForObjectDetection.from_pretrained("google/owlv2-base-patch16-ensemble").to(device).eval()
        sproc = SamProcessor.from_pretrained("facebook/sam-vit-base")
        sam = SamModel.from_pretrained("facebook/sam-vit-base").to(device).eval()
    except Exception as e:  # noqa: BLE001
        log(f"can masks: could not load the models ({e})")
        return None
    try:
        return _detect(images, device, long_side, score, max_cans, log, cv2, nms, oproc, owl, sproc, sam)
    except Exception as e:  # noqa: BLE001
        import traceback

        log(f"can masks: detection failed ({e!r})" + chr(10) + traceback.format_exc())
        return None


def _shape_ok(m, box, min_fill=0.45, min_solidity=0.8):
    """A can's mask fills much of its box and is convex (a body, an ellipse, or the convex
    hull of both); a mask that spreads thinly or wraps around something is not a can."""
    import cv2

    x0, y0, x1, y1 = box
    area = m.sum()
    if area < 30:
        return False, 0.0, 0.0
    fill = area / max(1.0, (x1 - x0) * (y1 - y0))
    cs, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    hull = cv2.convexHull(np.concatenate(cs)) if cs else None
    sol = area / max(1.0, cv2.contourArea(hull)) if hull is not None else 0.0
    return bool(fill >= min_fill and sol >= min_solidity), float(fill), float(sol)


def _detect(images, device, long_side, score, max_cans, log, cv2, nms, oproc, owl, sproc, sam, min_rel_area=0.25):
    queries = CAN_QUERIES + OTHER_QUERIES
    masks, boxes_all, inst_all = [], [], []
    dropped_shape = 0
    for im in images:
        h, w = im.shape[:2]
        f = min(1.0, long_side / max(h, w))
        small = cv2.resize(im, (round(w * f), round(h * f)), interpolation=cv2.INTER_AREA) if f < 1 else im
        hs, ws = small.shape[:2]
        with torch.no_grad():
            inp = oproc(text=[queries], images=small, return_tensors="pt").to(device)
            out = owl(**inp)
            side = max(hs, ws)  # OWLv2 pads to a square (bottom, right)
            res = _post_process(oproc, out, score, torch.tensor([[side, side]], device=device))
        b, sc, lab = res["boxes"], res["scores"], res["labels"]
        keep = lab < len(CAN_QUERIES)
        b, sc = b[keep], sc[keep]
        if len(b):
            k = nms(b.float(), sc.float(), 0.5)[:max_cans]
            b = b[k]
            b[:, [0, 2]] = b[:, [0, 2]].clamp(0, ws - 1)
            b[:, [1, 3]] = b[:, [1, 3]].clamp(0, hs - 1)
            # drop boxes covering most of the image (the whole scene)
            area = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
            b = b[area < 0.5 * hs * ws]
            # and boxes much smaller than the largest can in the photo (a bottle neck, a cap
            # under the cans): a can hidden that much adds little and a wrong one misleads
            if len(b):
                area = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
                b = b[area >= min_rel_area * area.max()]
        inst = np.zeros((0, hs, ws), bool)
        if len(b):
            with torch.no_grad():
                sin = sproc(small, input_boxes=[b.cpu().tolist()], return_tensors="pt").to(device)
                so = sam(**sin, multimask_output=False)
                pp = getattr(sproc, "post_process_masks", None) or sproc.image_processor.post_process_masks
                rs = sin.get("reshaped_input_sizes", sin["original_sizes"])
                m = pp(so.pred_masks.cpu(), sin["original_sizes"].cpu(), rs.cpu())[0]
            m = m[:, 0].numpy().astype(bool)
            bn = b.cpu().numpy()
            keep_i = []
            for i, (x0, y0, x1, y1) in enumerate(bn):
                # SAM spreads into what the can stands on or leans against (the tablet, the
                # cardboard): a can lies inside its box, give or take a few pixels
                px, py = 0.04 * (x1 - x0) + 2, 0.04 * (y1 - y0) + 2
                clip = np.zeros_like(m[i])
                clip[max(0, int(y0 - py)):int(y1 + py) + 1, max(0, int(x0 - px)):int(x1 + px) + 1] = True
                m[i] &= clip
                ok, _, _ = _shape_ok(m[i], (x0, y0, x1, y1))
                if ok:
                    keep_i.append(i)
                else:
                    dropped_shape += 1
            inst, b = m[keep_i], b[keep_i]
        union = inst.any(0) if len(inst) else np.zeros((hs, ws), bool)
        if f < 1:
            union = cv2.resize(union.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST).astype(bool)
            inst = np.stack([cv2.resize(x.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST).astype(bool) for x in inst])                 if len(inst) else np.zeros((0, h, w), bool)
        masks.append(union)
        inst_all.append(inst)
        boxes_all.append(b.cpu().numpy() / f if len(b) else np.zeros((0, 4)))
    log(f"can masks: {sum(len(x) for x in boxes_all)} cans in {len(images)} photos "
        f"({', '.join(str(len(x)) for x in boxes_all)}); {dropped_shape} dropped by shape")
    return masks, boxes_all, inst_all


def sam_box_prompter(device="cpu", long_side=768, log=print):
    """A function (image HxWx3 uint8 RGB, boxes (N, 4) in its pixels) -> (N, H, W) bool masks
    from SAM prompted with those boxes; or None if SAM cannot be loaded. Used to segment a
    can the detector missed (most often one partly hidden behind another) at the box where
    the fitted scene says it is."""
    try:
        import cv2
        from transformers import SamModel, SamProcessor

        sproc = SamProcessor.from_pretrained("facebook/sam-vit-base")
        sam = SamModel.from_pretrained("facebook/sam-vit-base").to(device).eval()
    except Exception as e:  # noqa: BLE001
        log(f"can masks: SAM unavailable for prompting ({e})")
        return None

    def run(image, boxes):
        h, w = image.shape[:2]
        f = min(1.0, long_side / max(h, w))
        small = cv2.resize(image, (round(w * f), round(h * f)), interpolation=cv2.INTER_AREA) if f < 1 else image
        b = (np.asarray(boxes, float) * f).tolist()
        with torch.no_grad():
            sin = sproc(small, input_boxes=[b], return_tensors="pt").to(device)
            so = sam(**sin, multimask_output=False)
            pp = getattr(sproc, "post_process_masks", None) or sproc.image_processor.post_process_masks
            rs = sin.get("reshaped_input_sizes", sin["original_sizes"])
            m = pp(so.pred_masks.cpu(), sin["original_sizes"].cpu(), rs.cpu())[0][:, 0].numpy().astype(bool)
        if f < 1:
            m = np.stack([cv2.resize(x.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST).astype(bool) for x in m])
        return m

    return run
