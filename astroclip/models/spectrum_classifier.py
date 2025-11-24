from typing import Dict, Optional, Tuple, Union

import torch
import torch.nn as nn

from .astroclip import SpectrumHead


class SpectrumClassifier(nn.Module):
    """Wrap SpectrumHead with a linear classifier for multi-class tasks."""

    def __init__(
        self,
        model_path: str,
        num_classes: int = 6,
        embed_dim: int = 1024,
        classifier_dropout: float = 0.1,
        n_head: int = 4,
        model_embed_dim: int = 768,
        dropout: float = 0.1,
        freeze_backbone: bool = True,
        load_pretrained_weights: bool = True,
        use_stats_tokens: bool = False,
    ) -> None:
        super().__init__()
        self.spectrum_head = SpectrumHead(
            model_path=model_path,
            embed_dim=embed_dim,
            n_head=n_head,
            model_embed_dim=model_embed_dim,
            dropout=dropout,
            freeze_backbone=freeze_backbone,
            load_pretrained_weights=load_pretrained_weights,
            use_stats_tokens=use_stats_tokens,
        )
        self.classifier = nn.Sequential(
            nn.Dropout(classifier_dropout),
            nn.Linear(embed_dim, num_classes),
        )

    def forward(
        self,
        x: torch.Tensor,
        return_features: bool = False,
        return_attentions: bool = False,
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        features: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]] = (
            self.spectrum_head(x, return_weights=return_attentions)
        )

        attentions: Optional[torch.Tensor] = None
        if return_attentions:
            features, attentions = features  # type: ignore[assignment]

        logits = self.classifier(features)

        if return_features or return_attentions:
            output: Dict[str, torch.Tensor] = {"logits": logits, "features": features}
            if attentions is not None:
                output["attentions"] = attentions
            return output

        return logits
