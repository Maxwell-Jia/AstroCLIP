#!/usr/bin/env python3
"""Quick SpecFormer inference sanity check."""

import argparse
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from astroclip.env import format_with_env
from astroclip.models import SpecFormer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Load a pretrained SpecFormer checkpoint and run a forward pass."
    )
    parser.add_argument(
        "--checkpoint-path",
        type=str,
        help=(
            "Path to the SpecFormer Lightning checkpoint (.ckpt). Defaults to "
            "ASTROCLIP_ROOT/checkpoints/specformer/specformer.ckpt if it exists, "
            "otherwise falls back to the local repository path."
        ),
    )
    parser.add_argument(
        "--spectrum-npy",
        type=str,
        help="Optional .npy file containing a spectrum shaped [L] or [L, 1].",
    )
    parser.add_argument(
        "--sequence-length",
        type=int,
        default=7781,
        help="Length of a synthetic spectrum to generate when no file is provided.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device for inference, e.g. cpu or cuda.",
    )
    return parser.parse_args()


def import_dependencies():
    try:
        from astroclip.env import format_with_env
        from astroclip.models import SpecFormer
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "astroclip dependencies are missing. Install them first (e.g. pip install -e .) "
            "so SpecFormer and torchvision are available."
        ) from exc

    return format_with_env, SpecFormer


def resolve_checkpoint_path(path_arg: Optional[str], format_with_env) -> Path:
    if path_arg:
        return Path(path_arg).expanduser()

    env_candidate = Path(
        format_with_env("{ASTROCLIP_ROOT}/checkpoints/specformer/specformer.ckpt")
    )
    repo_candidate = (
        Path(__file__).resolve().parents[1]
        / "checkpoints"
        / "specformer"
        / "specformer.ckpt"
    )

    for candidate in (env_candidate, repo_candidate):
        if candidate.exists():
            return candidate

    return env_candidate


def load_specformer(
    checkpoint_path: Path, device: torch.device, specformer_cls
) -> torch.nn.Module:
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = specformer_cls(**checkpoint["hyper_parameters"])
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device)
    model.eval()
    return model


def prepare_spectrum(args: argparse.Namespace, device: torch.device) -> torch.Tensor:
    if args.spectrum_npy:
        spectrum = np.asarray(np.load(args.spectrum_npy), dtype=np.float32)
    else:
        x = np.linspace(0, 2 * np.pi, args.sequence_length, dtype=np.float32)
        spectrum = np.sin(x) + 0.05 * np.random.randn(args.sequence_length).astype(np.float32)

    if spectrum.ndim == 1:
        spectrum = spectrum[:, None]
    elif spectrum.ndim == 3 and spectrum.shape[0] == 1 and spectrum.shape[2] == 1:
        spectrum = spectrum[0]
    elif spectrum.ndim != 2:
        raise ValueError(f"Expected spectrum with 1 or 2 dims, got shape {spectrum.shape}")

    if spectrum.shape[1] != 1:
        raise ValueError(f"Expected spectrum second dimension to be 1, got shape {spectrum.shape}")

    tensor = torch.from_numpy(spectrum).unsqueeze(0).to(device)
    return tensor


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    # format_with_env, SpecFormer = import_dependencies()
    checkpoint_path = resolve_checkpoint_path(args.checkpoint_path, format_with_env)

    print(f"Loading SpecFormer checkpoint from: {checkpoint_path}")
    model = load_specformer(checkpoint_path, device, SpecFormer)

    spectrum = prepare_spectrum(args, device)
    print(f"Input spectrum tensor shape: {tuple(spectrum.shape)}")

    with torch.no_grad():
        outputs = model(spectrum)

    embedding = outputs["embedding"]
    reconstructions = outputs["reconstructions"]

    print(f"Embedding shape: {tuple(embedding.shape)}")
    print(f"Reconstruction shape: {tuple(reconstructions.shape)}")
    print(
        "First sequence reconstruction sample:",
        reconstructions[0, :5, 0].tolist(),
    )


if __name__ == "__main__":
    main()
