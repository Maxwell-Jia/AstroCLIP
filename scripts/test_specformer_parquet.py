#!/usr/bin/env python3
"""Run SpecFormer (and optional classifier head) on a sample from a parquet file."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from astroclip.data import interpolate_spectrum
from astroclip.env import format_with_env
from astroclip.models import SpecFormer, SpectrumClassifier


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Load a SpecFormer checkpoint and run inference on one spectrum from a parquet file."
    )
    parser.add_argument(
        "--parquet-path",
        type=str,
        default="data/train_spectra.parquet",
        help="Path to the parquet file containing LAMOST spectra.",
    )
    parser.add_argument(
        "--row-index",
        type=int,
        default=0,
        help="Row index to sample from the parquet file.",
    )
    parser.add_argument(
        "--checkpoint-path",
        type=str,
        help=(
            "Path to the SpecFormer checkpoint (.ckpt). Defaults to "
            "ASTROCLIP_ROOT/checkpoints/specformer/specformer.ckpt or the local repo path."
        ),
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device for inference, e.g. cpu or cuda.",
    )
    parser.add_argument(
        "--use-classifier",
        action="store_true",
        help="Also run the SpectrumClassifier (randomly initialized head) to verify end-to-end shapes.",
    )
    parser.add_argument(
        "--num-classes",
        type=int,
        default=6,
        help="Number of classes for the classifier head when --use-classifier is set.",
    )
    parser.add_argument(
        "--no-interpolate",
        action="store_true",
        help="Disable interpolation to 3800-9100 (7781 pts) before inference.",
    )
    parser.add_argument(
        "--use-stats-tokens",
        action="store_true",
        help="Keep mean/std tokens at start of sequence. Default off (tokens zeroed).",
    )
    return parser.parse_args()


def resolve_checkpoint_path(path_arg: str | None) -> Path:
    if path_arg:
        return Path(path_arg).expanduser()

    env_candidate = Path(
        format_with_env("{ASTROCLIP_ROOT}/checkpoints/specformer/specformer.ckpt")
    )
    repo_candidate = Path(__file__).resolve().parents[1] / "checkpoints" / "specformer" / "specformer.ckpt"

    for candidate in (env_candidate, repo_candidate):
        if candidate.exists():
            return candidate

    return env_candidate


def load_specformer(checkpoint_path: Path, device: torch.device, use_stats_tokens: bool) -> SpecFormer:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = SpecFormer(**checkpoint["hyper_parameters"])
    model.load_state_dict(checkpoint["state_dict"])
    model.use_stats_tokens = use_stats_tokens
    model.to(device).eval()
    return model


def prepare_sample(parquet_path: Path, row_index: int, device: torch.device, do_interpolate: bool):
    df = pd.read_parquet(parquet_path)
    if row_index < 0 or row_index >= len(df):
        raise IndexError(f"Row index {row_index} out of bounds for file with {len(df)} rows")

    row = df.iloc[row_index]
    flux = np.asarray(row["flux"], dtype=np.float32)
    if do_interpolate:
        wavelength = np.asarray(row["wavelength"], dtype=np.float32)
        flux = interpolate_spectrum(wavelength=wavelength, flux=flux)
    if flux.ndim == 1:
        flux = flux[:, None]

    tensor = torch.from_numpy(flux).unsqueeze(0).to(device)
    meta = {
        "spectrum_id": row.get("spectrum_id"),
        "label": row.get("label"),
        "label_id": row.get("label_id"),
        "redshift": row.get("redshift"),
    }
    return tensor, meta


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    checkpoint_path = resolve_checkpoint_path(args.checkpoint_path)

    print(f"Using checkpoint: {checkpoint_path}")
    model = load_specformer(checkpoint_path, device, use_stats_tokens=args.use_stats_tokens)

    spectrum, meta = prepare_sample(Path(args.parquet_path), args.row_index, device, not args.no_interpolate)
    print(f"Loaded spectrum from row {args.row_index} with meta: {meta}")
    print(f"Spectrum tensor shape: {tuple(spectrum.shape)}")

    with torch.no_grad():
        outputs = model(spectrum)

    embedding = outputs["embedding"]
    reconstructions = outputs["reconstructions"]
    print(f"SpecFormer embedding shape: {tuple(embedding.shape)}")
    print(f"SpecFormer reconstruction shape: {tuple(reconstructions.shape)}")
    print("First 5 reconstructed values of first sequence:", reconstructions[0, :5, 0].tolist())

    if args.use_classifier:
        classifier = SpectrumClassifier(
            model_path=str(checkpoint_path),
            num_classes=args.num_classes,
            freeze_backbone=True,
            use_stats_tokens=args.use_stats_tokens,
        ).to(device)
        classifier.eval()
        with torch.no_grad():
            logits = classifier(spectrum)
        print(f"SpectrumClassifier logits shape: {tuple(logits.shape)}")
        print("Sample logits:", logits[0].tolist())
        print(
            "Note: classifier head is randomly initialized; values are only for shape/debug checks."
        )


if __name__ == "__main__":
    main()
