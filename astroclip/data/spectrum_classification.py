from typing import Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset, random_split


def interpolate_spectrum(
    wavelength: np.ndarray,
    flux: np.ndarray,
    start: float = 3800.0,
    end: float = 9100.0,
    num: int = 7781,
) -> np.ndarray:
    """Linearly interpolate flux onto a fixed wavelength grid."""
    w = np.asarray(wavelength, dtype=np.float32).squeeze()
    f = np.asarray(flux, dtype=np.float32).squeeze()
    mask = np.isfinite(w) & np.isfinite(f)
    if mask.sum() < 2:
        raise ValueError("Not enough finite points to interpolate.")
    w, f = w[mask], f[mask]
    order = np.argsort(w)
    w, f = w[order], f[order]
    target_w = np.linspace(start, end, num, dtype=np.float32)
    interp = np.interp(target_w, w, f, left=f[0], right=f[-1])
    return interp.astype(np.float32)


class SpectrumClassificationDataset(Dataset):
    """Parquet-backed spectrum classification dataset for SpecFormer fine-tuning."""

    def __init__(
        self,
        parquet_path: str,
        spectrum_key: str = "flux",
        label_key: str = "label_id",
        label_name_key: Optional[str] = "label",
        dropna: bool = True,
        target_length: Optional[int] = None,
        padding_value: float = 0.0,
        interpolate: bool = True,
        wavelength_key: str = "wavelength",
        interp_start: float = 3800.0,
        interp_end: float = 9100.0,
        interp_num: int = 7781,
    ) -> None:
        super().__init__()
        self.df = pd.read_parquet(parquet_path)
        if dropna:
            self.df = self.df.dropna(subset=[label_key])
        self.spectrum_key = spectrum_key
        self.label_key = label_key
        self.label_name_key = label_name_key
        self.padding_value = padding_value
        self.interpolate = interpolate
        self.wavelength_key = wavelength_key
        self.interp_start = interp_start
        self.interp_end = interp_end
        self.interp_num = interp_num

        labels = self.df[self.label_key].astype(int).to_numpy()
        self.labels = labels
        self.num_classes = int(labels.max()) + 1

        if self.interpolate and self.wavelength_key not in self.df.columns:
            raise ValueError(
                f"interpolate=True but wavelength_key '{self.wavelength_key}' not in dataframe columns"
            )

        if self.label_name_key and self.label_name_key in self.df.columns:
            mapping = {}
            for lbl_id, lbl_name in self.df[[self.label_key, self.label_name_key]].dropna().itertuples(index=False):
                lbl_id = int(lbl_id)
                if lbl_id not in mapping:
                    mapping[lbl_id] = str(lbl_name)
            self.label_names = [mapping.get(i, str(i)) for i in range(self.num_classes)]
        else:
            self.label_names = [str(i) for i in range(self.num_classes)]

        if target_length is None:
            self.target_length = self.interp_num if self.interpolate else max(
                len(np.asarray(v)) for v in self.df[self.spectrum_key]
            )
        else:
            self.target_length = target_length

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        row = self.df.iloc[idx]
        flux = np.asarray(row[self.spectrum_key], dtype=np.float32)

        if self.interpolate:
            wavelength = np.asarray(row[self.wavelength_key], dtype=np.float32)
            flux = interpolate_spectrum(
                wavelength=wavelength,
                flux=flux,
                start=self.interp_start,
                end=self.interp_end,
                num=self.interp_num,
            )

        if flux.ndim == 1:
            flux = flux[:, None]

        if flux.shape[0] < self.target_length:
            pad_width = self.target_length - flux.shape[0]
            flux = np.pad(
                flux,
                pad_width=((0, pad_width), (0, 0)),
                mode="constant",
                constant_values=self.padding_value,
            )
        elif flux.shape[0] > self.target_length:
            flux = flux[: self.target_length]

        spectrum = torch.from_numpy(flux)
        label = torch.tensor(int(row[self.label_key]), dtype=torch.long)
        return {"spectrum": spectrum, "label": label}


class SpectrumClassificationDataModule:
    """Thin DataModule-like helper wrapping DataLoaders."""

    def __init__(
        self,
        parquet_path: str,
        test_parquet_path: Optional[str] = None,
        batch_size: int = 64,
        num_workers: int = 0,
        val_split: float = 0.1,
        seed: int = 42,
        pin_memory: bool = False,
        target_length: Optional[int] = None,
        padding_value: float = 0.0,
        interpolate: bool = True,
        wavelength_key: str = "wavelength",
        interp_start: float = 3800.0,
        interp_end: float = 9100.0,
        interp_num: int = 7781,
        label_name_key: Optional[str] = "label",
    ) -> None:
        self.dataset = SpectrumClassificationDataset(
            parquet_path=parquet_path,
            target_length=target_length,
            padding_value=padding_value,
            interpolate=interpolate,
            wavelength_key=wavelength_key,
            interp_start=interp_start,
            interp_end=interp_end,
            interp_num=interp_num,
            label_name_key=label_name_key,
        )
        self.test_parquet_path = test_parquet_path
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.val_split = val_split
        self.seed = seed
        self.pin_memory = pin_memory
        self.train_ds: Optional[Dataset] = None
        self.val_ds: Optional[Dataset] = None
        self.test_ds: Optional[Dataset] = None
        self._dataset_kwargs = dict(
            label_name_key=label_name_key,
            target_length=self.dataset.target_length,
            padding_value=padding_value,
            interpolate=interpolate,
            wavelength_key=wavelength_key,
            interp_start=interp_start,
            interp_end=interp_end,
            interp_num=interp_num,
        )
        self.label_names = self.dataset.label_names

    def setup(self) -> None:
        if self.val_split <= 0:
            self.train_ds = self.dataset
            self.val_ds = None
        else:
            val_size = max(1, int(len(self.dataset) * self.val_split))
            train_size = len(self.dataset) - val_size
            self.train_ds, self.val_ds = random_split(
                self.dataset,
                [train_size, val_size],
                generator=torch.Generator().manual_seed(self.seed),
            )

        if self.test_parquet_path:
            self.test_ds = SpectrumClassificationDataset(
                parquet_path=self.test_parquet_path,
                **self._dataset_kwargs,
            )

    def train_dataloader(self) -> DataLoader:
        if self.train_ds is None:
            self.setup()
        return DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )

    def val_dataloader(self) -> Optional[DataLoader]:
        if self.val_split <= 0:
            return None
        if self.val_ds is None:
            self.setup()
        return DataLoader(
            self.val_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )

    def test_dataloader(self) -> Optional[DataLoader]:
        if self.test_parquet_path is None:
            return None
        if self.test_ds is None:
            self.setup()
        return DataLoader(
            self.test_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )
