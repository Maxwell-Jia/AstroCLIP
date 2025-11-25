from typing import Any, Dict, Optional, Sequence

import lightning as L
import torch
import torch.nn.functional as F
from torchmetrics.classification import (
    MulticlassAccuracy,
    MulticlassF1Score,
    MulticlassPrecision,
    MulticlassRecall,
)

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
        label_names: Optional[Sequence[str]] = None,
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
        self.num_classes = num_classes
        self.label_names = list(label_names) if label_names is not None else None

        # Validation metrics
        self.val_acc = MulticlassAccuracy(num_classes=num_classes)
        self.val_precision = MulticlassPrecision(num_classes=num_classes, average=None)
        self.val_recall = MulticlassRecall(num_classes=num_classes, average=None)
        self.val_f1 = MulticlassF1Score(num_classes=num_classes, average=None)
        self.test_acc = MulticlassAccuracy(num_classes=num_classes)
        self.test_precision = MulticlassPrecision(num_classes=num_classes, average=None)
        self.test_recall = MulticlassRecall(num_classes=num_classes, average=None)
        self.test_f1 = MulticlassF1Score(num_classes=num_classes, average=None)

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
        targets = batch["label"]

        # update metrics
        self.val_acc.update(logits, targets)
        self.val_precision.update(logits, targets)
        self.val_recall.update(logits, targets)
        self.val_f1.update(logits, targets)

        self.log("val_loss", loss, prog_bar=True, on_epoch=True, on_step=False)
        return loss

    def on_validation_epoch_end(self) -> None:
        acc = self.val_acc.compute()
        precision = self.val_precision.compute()
        recall = self.val_recall.compute()
        f1 = self.val_f1.compute()

        self.log("val_acc", acc, prog_bar=True, on_epoch=True)
        self.log("val_precision_macro", precision.mean(), prog_bar=False, on_epoch=True)
        self.log("val_recall_macro", recall.mean(), prog_bar=False, on_epoch=True)
        self.log("val_f1_macro", f1.mean(), prog_bar=False, on_epoch=True)

        # per-class metrics
        for i, (p, r, f) in enumerate(zip(precision, recall, f1)):
            label_tag = self._label_tag(i)
            self.log(f"val_precision_{label_tag}", p, prog_bar=False, on_epoch=True)
            self.log(f"val_recall_{label_tag}", r, prog_bar=False, on_epoch=True)
            self.log(f"val_f1_{label_tag}", f, prog_bar=False, on_epoch=True)

        # reset for next epoch
        self.val_acc.reset()
        self.val_precision.reset()
        self.val_recall.reset()
        self.val_f1.reset()

    def test_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        logits = self(batch["spectrum"])
        loss = F.cross_entropy(logits, batch["label"])
        targets = batch["label"]

        self.test_acc.update(logits, targets)
        self.test_precision.update(logits, targets)
        self.test_recall.update(logits, targets)
        self.test_f1.update(logits, targets)

        self.log("test_loss", loss, prog_bar=True, on_epoch=True, on_step=False)
        return loss

    def on_test_epoch_end(self) -> None:
        acc = self.test_acc.compute()
        precision = self.test_precision.compute()
        recall = self.test_recall.compute()
        f1 = self.test_f1.compute()

        self.log("test_acc", acc, prog_bar=True, on_epoch=True)
        self.log("test_precision_macro", precision.mean(), prog_bar=False, on_epoch=True)
        self.log("test_recall_macro", recall.mean(), prog_bar=False, on_epoch=True)
        self.log("test_f1_macro", f1.mean(), prog_bar=False, on_epoch=True)

        for i, (p, r, f) in enumerate(zip(precision, recall, f1)):
            label_tag = self._label_tag(i)
            self.log(f"test_precision_{label_tag}", p, prog_bar=False, on_epoch=True)
            self.log(f"test_recall_{label_tag}", r, prog_bar=False, on_epoch=True)
            self.log(f"test_f1_{label_tag}", f, prog_bar=False, on_epoch=True)

        self.test_acc.reset()
        self.test_precision.reset()
        self.test_recall.reset()
        self.test_f1.reset()

    def _label_name(self, idx: int) -> str:
        if self.label_names and idx < len(self.label_names):
            return self.label_names[idx]
        return str(idx)

    def _label_tag(self, idx: int) -> str:
        name = self._label_name(idx)
        safe = name.lower().replace(" ", "_").replace("/", "_")
        return f"class_{safe}"

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )
        return optimizer
