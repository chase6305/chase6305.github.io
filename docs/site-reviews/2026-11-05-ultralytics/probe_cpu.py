#!/usr/bin/env python3
"""Optional CPU source probes for the pinned Ultralytics article.

Use an isolated environment with PyTorch and Ultralytics' dependencies. The
script imports a read-only source checkout; it never installs packages or fetches
weights/data. All settings, synthetic images and model outputs go under --output.
ONNX/ORT checks additionally need the versions documented in source-results.json.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

REVISION = "1a767da5e58467ab8fbc2407b600b7632033dbc5"
sys.dont_write_bytecode = True


def configure(args):
    revision = subprocess.check_output(
        ["git", "-C", str(args.source_root), "rev-parse", "HEAD"], text=True).strip()
    if revision != REVISION:
        raise ValueError(f"Expected checkout {REVISION}, found {revision}")
    os.environ.update(YOLO_CONFIG_DIR=str(args.output / "config"),
                      MPLCONFIGDIR=str(args.output / "mpl"),
                      YOLO_OFFLINE="true", YOLO_AUTOINSTALL="false",
                      OMP_NUM_THREADS="2", MKL_NUM_THREADS="2")
    sys.path.insert(0, str(args.source_root))
    import torch
    import ultralytics
    from ultralytics.utils import SETTINGS, USER_CONFIG_DIR
    assert ultralytics.__version__ == "8.4.202"
    SETTINGS.update({key: False for key in (
        "sync", "clearml", "comet", "dvc", "mlflow", "raytune", "tensorboard", "wandb")})
    # Avoid font download checks. No plots are enabled in the training probe.
    import matplotlib.font_manager as font_manager
    font = font_manager.findfont("DejaVu Sans")
    USER_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    if not (USER_CONFIG_DIR / "Arial.ttf").exists():
        shutil.copyfile(font, USER_CONFIG_DIR / "Arial.ttf")
    torch.set_num_threads(2)
    torch.manual_seed(0)
    return torch, ultralytics


def geometry_and_gradients(torch, ultralytics):
    import numpy as np
    from ultralytics.data.augment import LetterBox
    from ultralytics.nn.modules.head import Detect
    from ultralytics.utils.loss import BboxLoss
    from ultralytics.utils.nms import non_max_suppression
    from ultralytics.utils.ops import scale_boxes
    from ultralytics.utils.tal import TaskAlignedAssigner
    image = np.zeros((720, 1280, 3), dtype=np.uint8)
    square = LetterBox(new_shape=(640, 640), auto=False)(image=image)
    minimal = LetterBox(new_shape=(640, 640), auto=True, stride=32)(image=image)
    assert square.shape == (640, 640, 3) and minimal.shape == (384, 640, 3)
    assert np.all(square[:140] == 114) and np.all(minimal[:12] == 114)
    restored = scale_boxes((640, 640), torch.tensor([[160., 230., 480., 410.]]), (720, 1280))
    torch.testing.assert_close(restored, torch.tensor([[320., 180., 960., 540.]]))
    repeated = torch.tensor([[[10., 10., 30., 30., .9, 0.], [10., 10., 30., 30., .8, 0.]]])
    assert len(non_max_suppression(repeated, .25, .1, end2end=True)[0]) == 2

    head = Detect(nc=2, reg_max=1, end2end=True, ch=(8, 16, 32)).train()
    features = [torch.randn(2, c, size, size, requires_grad=True)
                for c, size in ((8, 8), (16, 4), (32, 2))]
    predictions = head(features)
    one = predictions["one2one"]
    own_parameters = list(head.one2one_cv3.parameters())
    gradients = torch.autograd.grad(one["boxes"].square().mean() + one["scores"].square().mean(),
                                    features + own_parameters, allow_unused=True, retain_graph=True)
    assert all(value is None for value in gradients[:3])
    assert any(value is not None and value.abs().max() > 0 for value in gradients[3:])
    many = predictions["one2many"]
    backbone = torch.autograd.grad(many["boxes"].square().mean() + many["scores"].square().mean(),
                                   features, allow_unused=True)
    assert all(value is not None and value.abs().max() > 0 for value in backbone)
    overlap, distance = BboxLoss(reg_max=1)(
        torch.full((1, 1, 4), 2.), torch.tensor([[[2., 2., 6., 6.]]]), torch.tensor([[4., 4.]]),
        torch.tensor([[[1., 1., 7., 7.]]]), torch.ones(1, 1, 1), torch.tensor(1.),
        torch.tensor([[True]]), torch.tensor([64., 64.]), torch.tensor([[8.]]))
    # Independent concentric-square areas and normalized edge distances.
    torch.testing.assert_close(overlap, torch.tensor(1 - 16/36))
    torch.testing.assert_close(distance, torch.tensor(8/64))
    assigner = TaskAlignedAssigner(num_classes=2, stride=[8, 16, 32])
    candidates = assigner.select_candidates_in_gts(
        torch.tensor([[4., 4.], [12., 12.], [24., 24.]]),
        torch.tensor([[[7., 7., 9., 9.]]]), torch.ones(1, 1, 1))
    assert candidates.tolist() == [[[True, True, False]]]

    cases = []
    for yaml, mode in (("yolo11n.yaml", None), ("yolo26n.yaml", None), ("yolo26n.yaml", False)):
        model = ultralytics.YOLO(yaml)
        result = model.predict(image, imgsz=640, rect=False, device="cpu", nms=mode,
                               conf=.00001, max_det=5, verbose=False)[0]
        assert result.orig_shape == (720, 1280) and result.boxes is not None
        assert torch.isfinite(result.boxes.data).all()
        effective = model.predictor.model.end2end
        assert effective == (mode is False)
        cases.append(dict(model=yaml, nms=mode, end2end=effective,
                          reg_max=model.model.model[-1].reg_max))
    return dict(square_wh=[640, 640], minimal_wh=[640, 384],
                one2one_feature_gradient="detached", one2one_head_gradient="present",
                one2many_feature_gradient="present", ciou_loss=float(overlap),
                l1_loss=float(distance), tiny_candidate_mask=candidates.tolist(), models=cases)


def make_data(root):
    from PIL import Image, ImageDraw
    for split, count in (("train", 8), ("val", 4)):
        for kind in ("images", "labels"):
            (root / "data" / kind / split).mkdir(parents=True, exist_ok=True)
        for index in range(count):
            image = Image.new("RGB", (96, 64), (30+index*3, 30, 30))
            x, y = 8+(index % 4)*10, 8+(index % 2)*8
            ImageDraw.Draw(image).rectangle((x, y, x+23, y+23),
                                            fill=(200, 80, 40) if index % 2 else (40, 180, 210))
            image.save(root / "data/images" / split / f"{index:02d}.png")
            (root / "data/labels" / split / f"{index:02d}.txt").write_text(
                f"{index % 2} {(2*x+24)/192} {(2*y+24)/128} {24/96} {24/64}\n")
    (root / "data/toy.yaml").write_text(
        f"path: {root}/data\ntrain: images/train\nval: images/val\nnames:\n  0: cyan_part\n  1: orange_part\n")


def train_phase(args, torch, ultralytics):
    checkpoint = args.output / "runs/recovery/weights/last.pt"
    if args.phase == "interrupt":
        make_data(args.output)
        model = ultralytics.YOLO("yolo26n.yaml")
        class Interrupted(RuntimeError):
            pass
        def interrupt(trainer):
            if trainer.epoch == 0:
                raise Interrupted("intentional interruption after checkpoint")
        model.add_callback("on_fit_epoch_end", interrupt)
        try:
            model.train(data=str(args.output / "data/toy.yaml"), epochs=3, imgsz=64,
                        batch=2, device="cpu", workers=0, pretrained=False, amp=False,
                        plots=False, optimizer="SGD", lr0=.001, nbs=2, warmup_epochs=0,
                        project=str(args.output / "runs"), name="recovery", save_period=1,
                        close_mosaic=0, mosaic=0, mixup=0, scale=0, translate=0,
                        fliplr=0, hsv_h=0, hsv_s=0, hsv_v=0, nms=False, max_det=10)
        except Interrupted:
            pass
        state = torch.load(checkpoint, map_location="cpu", weights_only=False)
        assert state["epoch"] == 0 and state["optimizer"] is not None
    else:
        model = ultralytics.YOLO(checkpoint)
        model.train(resume=True)
        assert model.trainer.start_epoch == 1
        state = torch.load(checkpoint, map_location="cpu", weights_only=False)
        assert state["epoch"] == -1 and state["optimizer"] is None


def check_onnx(args, torch, ultralytics):
    import cv2
    import numpy as np
    import onnx
    import onnxruntime as ort
    from ultralytics.data.augment import LetterBox
    from ultralytics.utils import ASSETS
    image = cv2.imread(str(ASSETS / "bus.jpg"))
    padded = LetterBox(new_shape=(640, 640), auto=False)(image=image)
    array = np.ascontiguousarray(padded[:, :, ::-1].transpose(2, 0, 1)[None]).astype(np.float32)/255
    results = []
    for name, mode in (("standard", None), ("e2e", False)):
        weight = args.output / f"{name}.pt"
        shutil.copyfile(args.weights, weight)
        model = ultralytics.YOLO(weight)
        exported = model.export(format="onnx", imgsz=640, dynamic=False,
                                simplify=True, opset=17, device="cpu", nms=mode)
        onnx.checker.check_model(onnx.load(exported))
        options = ort.SessionOptions()
        options.intra_op_num_threads, options.inter_op_num_threads = 2, 1
        session = ort.InferenceSession(str(exported), sess_options=options, providers=["CPUExecutionProvider"])
        actual = session.run(None, {session.get_inputs()[0].name: array})[0]
        model.model.eval()
        model.model.end2end = mode is False
        model.model.fuse(verbose=False)
        with torch.no_grad():
            expected = model.model(torch.from_numpy(array))[0].numpy()
        assert expected.shape == actual.shape
        if mode is False:
            actual, expected = actual[0], expected[0]
            actual, expected = actual[actual[:, 4] > .25], expected[expected[:, 4] > .25]
        np.testing.assert_allclose(actual, expected, rtol=1e-3, atol=1e-3)
        wrapped = ultralytics.YOLO(exported, task="detect").predict(
            ASSETS/"bus.jpg", imgsz=640, rect=False, device="cpu", conf=.25, verbose=False)[0]
        native = ultralytics.YOLO(weight).predict(
            ASSETS/"bus.jpg", imgsz=640, rect=False, device="cpu", conf=.25, nms=mode, verbose=False)[0]
        np.testing.assert_allclose(wrapped.boxes.data.numpy(), native.boxes.data.numpy(), rtol=1e-3, atol=2e-3)
        results.append(dict(name=name, nms=mode, shape=list(session.get_outputs()[0].shape),
                            max_absolute_difference=float(np.abs(actual-expected).max()),
                            retained_boxes=len(wrapped.boxes)))
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="new disposable output directory")
    parser.add_argument("--weights", type=Path, help="existing trusted YOLO26n weights; never downloaded")
    parser.add_argument("--training", action="store_true")
    parser.add_argument("--onnx", action="store_true")
    parser.add_argument("--phase", choices=("main", "interrupt", "resume"), default="main", help=argparse.SUPPRESS)
    args = parser.parse_args()
    args.source_root, args.output = args.source_root.resolve(), args.output.resolve()
    if args.onnx and (args.weights is None or not args.weights.is_file()):
        parser.error("--onnx requires --weights pointing to an existing file")
    if args.phase == "main":
        args.output.mkdir(parents=True, exist_ok=False)
    torch, ultralytics = configure(args)
    if args.phase != "main":
        train_phase(args, torch, ultralytics)
        return
    report = dict(source_revision=REVISION, version=ultralytics.__version__,
                  torch=torch.__version__, source_file=ultralytics.__file__,
                  geometry_and_gradients=geometry_and_gradients(torch, ultralytics))
    if args.training:
        for phase in ("interrupt", "resume"):
            command = [sys.executable, "-B", __file__, "--source-root", str(args.source_root),
                       "--output", str(args.output), "--phase", phase]
            process = subprocess.run(command, capture_output=True, text=True, timeout=180)
            (args.output / f"training-{phase}.log").write_text(process.stdout + process.stderr)
            process.check_returncode()
        report["training"] = "checkpoint interrupted after epoch 1, resumed at epoch 2 in a fresh process; completed state stripped"
    if args.onnx:
        report["weight_sha256"] = hashlib.sha256(args.weights.read_bytes()).hexdigest()
        report["onnx"] = check_onnx(args, torch, ultralytics)
    report["limitations"] = "CPU implementation probes; synthetic training is not accuracy evaluation; one-image ONNX comparison is not full dataset/backend validation"
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
