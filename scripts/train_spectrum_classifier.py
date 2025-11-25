#!/usr/bin/env python3
"""Fine-tune SpecFormer + SpectrumHead with a 6-class classifier on parquet spectra."""

import argparse
from pathlib import Path

import lightning as L
import torch

from astroclip.data import SpectrumClassificationDataModule
from astroclip.env import format_with_env
from astroclip.models import SpectrumClassifierModule


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train spectrum classifier.")
    parser.add_argument(
        "--parquet-path",
        type=str,
        default="data/train_spectra.parquet",
        help="Parquet file with columns flux (np.ndarray) and label_id (int).",
    )
    parser.add_argument(
        "--checkpoint-path",
        type=str,
        help="SpecFormer checkpoint path; defaults to ASTROCLIP_ROOT/checkpoints/specformer/specformer.ckpt.",
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="Dataloader workers; use 0 if multiprocessing is restricted.",
    )
    parser.add_argument(
        "--test-parquet-path",
        type=str,
        default="data/test_spectra.parquet",
        help="Held-out test parquet; columns match training parquet.",
    )
    parser.add_argument(
        "--val-split",
        type=float,
        default=0.0,
        help="Fraction of training set used for validation. Default 0 disables split.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--max-epochs", type=int, default=50)
    parser.add_argument("--devices", type=int, default=0, help="Number of devices (e.g. GPUs) to use.")
    parser.add_argument("--freeze-backbone", action="store_true", help="Freeze SpecFormer backbone weights.")
    parser.add_argument("--classifier-dropout", type=float, default=0.1)
    parser.add_argument("--fast-dev-run", action="store_true", help="Lightning fast_dev_run for smoke test.")
    parser.add_argument(
        "--use-stats-tokens",
        action="store_true",
        help="If set, keep mean/std tokens at the start of the sequence. Default is disabled (tokens zeroed).",
    )
    parser.add_argument(
        "--target-length",
        type=int,
        default=None,
        help="Pad/truncate spectra to this length. Defaults to max length found in the parquet.",
    )
    parser.add_argument(
        "--padding-value",
        type=float,
        default=0.0,
        help="Value used when padding spectra shorter than target-length.",
    )
    parser.add_argument(
        "--no-interpolate",
        action="store_true",
        help="Disable interpolation to fixed wavelength grid (3800-9100, 7781 points).",
    )
    parser.add_argument(
        "--wavelength-key",
        type=str,
        default="wavelength",
        help="Column name for wavelength when --interpolate is set.",
    )
    parser.add_argument(
        "--interp-start",
        type=float,
        default=3800.0,
        help="Interpolation start wavelength.",
    )
    parser.add_argument(
        "--interp-end",
        type=float,
        default=9100.0,
        help="Interpolation end wavelength.",
    )
    parser.add_argument(
        "--interp-num",
        type=int,
        default=7781,
        help="Number of points for interpolation grid.",
    )
    parser.add_argument(
        "--label-name-key",
        type=str,
        default="label",
        help="Column containing human-readable label names; defaults to 'label'.",
    )
    return parser.parse_args()


def resolve_checkpoint(path_arg: str | None) -> Path:
    if path_arg:
        return Path(path_arg).expanduser()
    env_candidate = Path(format_with_env("{ASTROCLIP_ROOT}/checkpoints/specformer/specformer.ckpt"))
    repo_candidate = Path(__file__).resolve().parents[1] / "checkpoints" / "specformer" / "specformer.ckpt"
    for candidate in (env_candidate, repo_candidate):
        if candidate.exists():
            return candidate
    return env_candidate


def main() -> None:
    args = parse_args()
    checkpoint_path = resolve_checkpoint(args.checkpoint_path)
    dm = SpectrumClassificationDataModule(
        parquet_path=args.parquet_path,
        test_parquet_path=args.test_parquet_path,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        val_split=args.val_split,
        seed=args.seed,
        pin_memory=torch.cuda.is_available(),
        target_length=args.target_length,
        padding_value=args.padding_value,
        interpolate=not args.no_interpolate,
        wavelength_key=args.wavelength_key,
        interp_start=args.interp_start,
        interp_end=args.interp_end,
        interp_num=args.interp_num,
        label_name_key=args.label_name_key,
    )
    dm.setup()

    num_classes = dm.dataset.num_classes
    model = SpectrumClassifierModule(
        model_path=str(checkpoint_path),
        num_classes=num_classes,
        lr=args.lr,
        weight_decay=args.weight_decay,
        freeze_backbone=args.freeze_backbone,
        classifier_dropout=args.classifier_dropout,
        use_stats_tokens=args.use_stats_tokens,
        label_names=dm.label_names,
    )

    trainer = L.Trainer(
        max_epochs=args.max_epochs,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=args.devices,
        fast_dev_run=args.fast_dev_run,
        deterministic=True,
        log_every_n_steps=10,
    )
    val_loader = dm.test_dataloader()
    trainer.fit(
        model,
        train_dataloaders=dm.train_dataloader(),
        val_dataloaders=val_loader if val_loader is not None else None,
    )


if __name__ == "__main__":
    main()
