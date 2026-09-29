"""Train a YOLO lamp detector from 5-column box labels."""
from __future__ import annotations

import argparse
from pathlib import Path


EXTERNAL_BACKBONE_CONFIGS = {
    "convnext_tiny": "configs/yolo_convnext_tiny_p2_underwater.yaml",
    "efficientnet_v2_s": "configs/yolo_efficientnet_v2_s_p2_underwater.yaml",
    "fastervit0": "configs/yolo_fastervit0_p2_underwater.yaml",
}

YOLO_VARIANT_CONFIGS = {
    (True, False, False): "configs/yolo11s_p2_underwater.yaml",
    (True, True, False): "configs/yolo11s_p2_attn_underwater.yaml",
    (True, False, True): "configs/yolo11s_p2_arf_underwater.yaml",
    (True, True, True): "configs/yolo11s_p2_attn_arf_underwater.yaml",
}

YOLO_P2_CHANNELS = {
    "s": 64,
    "m": 128,
    "l": 128,
}

YOLO_P2_CONCAT_CHANNELS = {
    "s": 256,
    "m": 512,
    "l": 512,
}


def _write_yolo_variant(
    project_root: Path,
    scale: str,
    *,
    attention: bool,
    arf: bool,
    attention_type: str = "light",
    fusion: str = "concat",
) -> Path:
    """Materialize a correctly scaled YOLO11 P2 ablation config."""
    import yaml

    precision_branch = fusion == "gated" or (attention and attention_type == "coordinate")
    if precision_branch:
        source_path = project_root / "configs/yolo11s_p2_precision_underwater.yaml"
    else:
        source_path = project_root / YOLO_VARIANT_CONFIGS[(True, attention, arf)]
    config = yaml.safe_load(source_path.read_text(encoding="utf-8"))
    config["scale"] = scale
    p2_channels = YOLO_P2_CHANNELS[scale]
    concat_channels = YOLO_P2_CONCAT_CHANNELS[scale]
    for layer in config["backbone"] + config["head"]:
        module_name = layer[2]
        if module_name in ("UnderwaterLightAttention", "AdaptiveReceptiveFieldFusion"):
            layer[3][0] = p2_channels
        elif module_name == "UnderwaterCrossScaleGate":
            layer[3] = [concat_channels, 8, fusion == "gated"]
        elif module_name == "ResidualCoordinateAttention":
            layer[3] = [p2_channels, 16, attention and attention_type == "coordinate"]

    generated_dir = project_root / "models" / "generated_configs"
    generated_dir.mkdir(parents=True, exist_ok=True)
    output_path = generated_dir / (
        f"yolo11{scale}_p2_fusion-{fusion}_attn-{attention_type if attention else 'none'}"
        f"_arf{int(arf)}.yaml"
    )
    output_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return output_path


def _write_external_variant(
    project_root: Path,
    backbone: str,
    *,
    attention: bool,
    arf: bool,
    imagenet_normalize: bool,
    imagenet_pretrained: bool,
) -> Path:
    """Materialize the exact external-backbone ablation config used by a run."""
    import yaml

    source_path = project_root / EXTERNAL_BACKBONE_CONFIGS[backbone]
    config = yaml.safe_load(source_path.read_text(encoding="utf-8"))
    enhancer_found = False
    torchvision_found = False

    for layer in config["backbone"] + config["head"]:
        module_name = layer[2]
        if module_name == "ImageNetNormalize":
            layer[3] = [imagenet_normalize]
        elif module_name == "TorchVision":
            layer[3][2] = "DEFAULT" if imagenet_pretrained else None
            torchvision_found = True
        elif module_name == "UnderwaterP2Enhancer":
            channels = layer[3][0]
            layer[3] = [channels, attention, arf]
            enhancer_found = True

    if not torchvision_found or not enhancer_found:
        raise RuntimeError(f"Backbone template is incomplete: {source_path}")

    generated_dir = project_root / "models" / "generated_configs"
    generated_dir.mkdir(parents=True, exist_ok=True)
    suffix = (
        f"{backbone}_p2_attn{int(attention)}_arf{int(arf)}"
        f"_norm{int(imagenet_normalize)}_pre{int(imagenet_pretrained)}.yaml"
    )
    output_path = generated_dir / suffix
    output_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="configs/underwater_light_box.yaml")
    parser.add_argument("--model", default="", help="Optional YAML or checkpoint. When set, module switches are ignored.")
    parser.add_argument(
        "--backbone",
        choices=("yolo11", *EXTERNAL_BACKBONE_CONFIGS),
        default="yolo11",
        help="Backbone ablation. External backbones use the same P2-P5 YOLO neck.",
    )
    parser.add_argument(
        "--yolo-scale",
        choices=tuple(YOLO_P2_CHANNELS),
        default="s",
        help="YOLO11 capacity. Use m as the primary accuracy-oriented comparison.",
    )
    parser.add_argument("--p2", action=argparse.BooleanOptionalAction, default=True, help="Enable the P2/4 small-lamp head.")
    parser.add_argument("--attention", action=argparse.BooleanOptionalAction, default=True, help="Enable channel-spatial light-halo attention.")
    parser.add_argument(
        "--attention-type",
        choices=("light", "coordinate"),
        default="light",
        help="P2 attention ablation. coordinate preserves directional position information.",
    )
    parser.add_argument(
        "--fusion",
        choices=("concat", "gated"),
        default="concat",
        help="P2 semantic/detail fusion. gated learns a per-image, per-channel balance.",
    )
    parser.add_argument("--arf", action=argparse.BooleanOptionalAction, default=False, help="Enable adaptive receptive-field fusion on P2.")
    parser.add_argument(
        "--imagenet-normalize",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use ImageNet mean/std for an external ImageNet-pretrained backbone.",
    )
    parser.add_argument(
        "--pretrained",
        default="auto",
        help="auto uses yolo11s.pt or external ImageNet weights; use none to train from scratch.",
    )
    parser.add_argument("--epochs", type=int, default=240)
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--batch", type=int, default=3, help="P2 uses more VRAM than the standard three-scale YOLO11s head.")
    parser.add_argument(
        "--effective-batch",
        type=int,
        default=64,
        help="Nominal batch (Ultralytics nbs). Gradient accumulation is round(effective_batch / batch).",
    )
    parser.add_argument("--workers", type=int, default=0, help="DataLoader workers; 0 is safest on low-RAM Windows systems.")
    parser.add_argument(
        "--cache",
        choices=("none", "ram", "disk"),
        default="none",
        help="Use none by default. disk creates one .npy cache per image and can consume substantial storage.",
    )
    parser.add_argument("--plots", action="store_true", help="Generate batch mosaic plots. Disabled by default to avoid OpenCV RAM spikes.")
    parser.add_argument(
        "--augmentation",
        choices=("baseline", "underwater"),
        default="underwater",
        help="Use baseline augmentation for an ablation or underwater colour/scale augmentation for the proposed model.",
    )
    parser.add_argument("--device", default=0)
    parser.add_argument("--project", default="models/yolo_lamp_box")
    parser.add_argument("--name", default="yolo11s_box_1280")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--patience", type=int, default=None, help="Override profile early-stopping patience.")
    parser.add_argument("--lr0", type=float, default=0.001)
    parser.add_argument("--weight-decay", type=float, default=0.0005)
    parser.add_argument("--box-gain", type=float, default=7.5)
    parser.add_argument("--cls-gain", type=float, default=0.5)
    parser.add_argument("--dfl-gain", type=float, default=1.5)
    args = parser.parse_args()

    if args.cache == "disk":
        print("WARNING: --cache disk creates .npy files beside every training image. Use --cache none to avoid them.")

    if args.batch < 1:
        parser.error("--batch must be a positive integer")
    if args.effective_batch < args.batch:
        parser.error("--effective-batch must be greater than or equal to --batch")
    if not args.p2 and (args.attention or args.arf):
        parser.error("--attention and --arf require --p2")
    if not args.p2 and args.fusion != "concat":
        parser.error("--fusion gated requires --p2")
    if not args.model and args.backbone != "yolo11" and not args.p2:
        parser.error("External backbone comparisons currently use the shared P2-P5 neck; keep --p2 enabled.")
    if not args.model and args.backbone != "yolo11" and args.yolo_scale != "s":
        parser.error("--yolo-scale applies only to --backbone yolo11.")
    if not args.model and args.backbone != "yolo11" and (
        args.fusion != "concat" or args.attention_type != "light"
    ):
        parser.error("--fusion gated and --attention-type coordinate currently apply only to YOLO11.")
    if args.arf and (args.fusion == "gated" or args.attention_type == "coordinate"):
        parser.error("Do not combine ARF with the precision P2 branch; use --no-arf for a clean ablation.")
    if not args.model and args.backbone == "fastervit0" and args.imgsz != 1280:
        parser.error("The FasterViT-0 detector adapter requires --imgsz 1280.")
    if min(args.box_gain, args.cls_gain, args.dfl_gain) <= 0:
        parser.error("Loss gains must be positive.")

    project_root = Path(__file__).resolve().parents[1]
    data_path = Path(args.data)
    output_root = Path(args.project)
    if not data_path.is_absolute():
        data_path = project_root / data_path
    if not output_root.is_absolute():
        output_root = project_root / output_root

    from .yolo_extensions import register_ultralytics_layers

    register_ultralytics_layers()
    from ultralytics import YOLO

    pretrained_arg = args.pretrained.strip()
    pretrained_mode = pretrained_arg.lower()
    external_pretrained = pretrained_mode in ("auto", "default", "imagenet")
    if args.backbone != "yolo11" and pretrained_mode not in ("auto", "default", "imagenet", "none", ""):
        parser.error("External backbones support --pretrained auto|imagenet|none.")

    if args.model:
        model_spec = args.model
        yolo_initialization = None if pretrained_mode in ("auto", "none", "") else pretrained_arg
    elif args.backbone == "yolo11":
        if args.p2:
            model_spec = _write_yolo_variant(
                project_root,
                args.yolo_scale,
                attention=args.attention,
                arf=args.arf,
                attention_type=args.attention_type,
                fusion=args.fusion,
            )
        else:
            model_spec = f"yolo11{args.yolo_scale}.yaml"
        yolo_initialization = f"yolo11{args.yolo_scale}.pt" if pretrained_mode == "auto" else pretrained_arg
    else:
        model_spec = _write_external_variant(
            project_root,
            args.backbone,
            attention=args.attention,
            arf=args.arf,
            imagenet_normalize=args.imagenet_normalize,
            imagenet_pretrained=external_pretrained,
        )
        yolo_initialization = None

    accumulation = max(round(args.effective_batch / args.batch), 1)
    actual_effective_batch = args.batch * accumulation
    print(
        "Model switches: "
        f"backbone={args.backbone}, yolo_scale={args.yolo_scale}, p2={args.p2}, "
        f"attention={args.attention}, attention_type={args.attention_type}, "
        f"fusion={args.fusion}, arf={args.arf}, "
        f"imagenet_norm={args.imagenet_normalize}; model={model_spec}"
    )
    print(
        f"Batching: micro_batch={args.batch}, requested_effective_batch={args.effective_batch}, "
        f"accumulate={accumulation}, actual_effective_batch={actual_effective_batch}"
    )
    model_path = Path(model_spec)
    if not model_path.is_absolute() and (project_root / model_path).exists():
        model_path = project_root / model_path
    model = YOLO(str(model_path))

    # A P2 YAML has extra layers, but its backbone and shared head layers can
    # still inherit COCO initialization from yolo11s.pt.
    if yolo_initialization and pretrained_mode != "none":
        model.load(yolo_initialization)

    augmentations = {
        "baseline": {
            "patience": 40,
            "warmup_epochs": 3.0,
            "close_mosaic": 20,
            "mosaic": 0.15,
            "mixup": 0.0,
            "degrees": 4.0,
            "translate": 0.04,
            "scale": 0.16,
            "fliplr": 0.0,
            "hsv_h": 0.01,
            "hsv_s": 0.18,
            "hsv_v": 0.18,
        },
        "underwater": {
            "patience": 60,
            "warmup_epochs": 5.0,
            "close_mosaic": 30,
            # Moderate underwater domain augmentation. Detection is invariant
            # to mirror direction; identity and GNN labels are not used here.
            "mosaic": 0.25,
            # MixUp allocates float image pairs in every DataLoader worker and
            # produces unrealistic overlapping light sources. Keep it off.
            "mixup": 0.0,
            "degrees": 3.0,
            "translate": 0.05,
            "scale": 0.25,
            "fliplr": 0.5,
            "hsv_h": 0.01,
            "hsv_s": 0.35,
            "hsv_v": 0.30,
        },
    }[args.augmentation]
    patience = args.patience if args.patience is not None else augmentations.pop("patience")

    trainer_class = None
    if not args.model and args.backbone == "fastervit0":
        from .yolo_trainers import SquareValidationDetectionTrainer

        trainer_class = SquareValidationDetectionTrainer

    model.train(
        trainer=trainer_class,
        data=str(data_path),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project=str(output_root),
        name=args.name,
        pretrained=True,
        optimizer="AdamW",
        lr0=args.lr0,
        lrf=0.01,
        cos_lr=True,
        nbs=args.effective_batch,
        weight_decay=args.weight_decay,
        box=args.box_gain,
        cls=args.cls_gain,
        dfl=args.dfl_gain,
        patience=patience,
        erasing=0.0,
        plots=args.plots,
        cache=False if args.cache == "none" else args.cache,
        workers=args.workers,
        seed=42,
        deterministic=True,
        save_period=20,
        resume=args.resume,
        **augmentations,
    )


if __name__ == "__main__":
    main()
