from typing import Any, Dict, Optional

import lightning as L
import torch
import torch.nn.functional as F

from .spectrum_classifier import SpectrumClassifier


class SpectrumClassifierModule(L.LightningModule):
    """LightningModule wrapper to fine-tune SpectrumClassifier."""

    def __init__(
        self,
        model_path: str,
        num_classes: int,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        freeze_backbone: bool = True,
        classifier_dropout: float = 0.1,
        use_stats_tokens: bool = False,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.model = SpectrumClassifier(
            model_path=model_path,
            num_classes=num_classes,
            freeze_backbone=freeze_backbone,
            classifier_dropout=classifier_dropout,
            use_stats_tokens=use_stats_tokens,
        )
        self.lr = lr
        self.weight_decay = weight_decay

    def forward(self, spectrum: torch.Tensor) -> torch.Tensor:
        return self.model(spectrum)

    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        logits = self(batch["spectrum"])
        loss = F.cross_entropy(logits, batch["label"])
        acc = (logits.argmax(dim=1) == batch["label"]).float().mean()
        self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        self.log("train_acc", acc, prog_bar=True, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> Optional[torch.Tensor]:
        logits = self(batch["spectrum"])
        loss = F.cross_entropy(logits, batch["label"])
        acc = (logits.argmax(dim=1) == batch["label"]).float().mean()
        self.log("val_loss", loss, prog_bar=True, on_epoch=True)
        self.log("val_acc", acc, prog_bar=True, on_epoch=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )
        return optimizer
