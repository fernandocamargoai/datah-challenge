# Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License").
# You may not use this file except in compliance with the License.
# A copy of the License is located at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# or in the "license" file accompanying this file. This file is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either
# express or implied. See the License for the specific language governing
# permissions and limitations under the License.
from typing import Dict, Tuple

import pytorch_lightning as pl
import torch

from gluonts.torch.modules.loss import DistributionLoss, NegativeLogLikelihood
from gluonts.torch.util import weighted_average

from .module import CausalDeepARModel


class CausalDeepARLightningModule(pl.LightningModule):
    def __init__(
        self,
        model: CausalDeepARModel,
        loss: DistributionLoss = NegativeLogLikelihood(),
        lr: float = 1e-3,
        weight_decay: float = 1e-8,
        control_loss_weight: float = 1.0,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.model = model
        self.loss = loss
        self.lr = lr
        self.weight_decay = weight_decay
        self.control_loss_weight = control_loss_weight

    def _compute_loss(self, batch: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        feat_static_cat = batch["feat_static_cat"]
        feat_static_real = batch["feat_static_real"]
        past_time_feat = batch["past_time_feat"]
        past_target = batch["past_target"]
        future_time_feat = batch["future_time_feat"]
        future_target = batch["future_target"]
        past_observed_values = batch["past_observed_values"]
        future_observed_values = batch["future_observed_values"]
        past_control = batch["past_control"]
        future_control = batch["future_control"]

        control_distr, distr = self.model.distribution(
            feat_static_cat=feat_static_cat,
            feat_static_real=feat_static_real,
            past_time_feat=past_time_feat,
            past_target=past_target,
            past_observed_values=past_observed_values,
            future_time_feat=future_time_feat,
            future_target=future_target,
            future_observed_values=future_observed_values,
            past_control=past_control,
            future_control=future_control,
        )

        context_target = past_target[:, -self.model.context_length + 1 :]
        target = torch.cat(
            (context_target, future_target),
            dim=1,
        )

        context_control = past_control[:, -self.model.context_length + 1:]
        control = torch.cat(
            (context_control, future_control),
            dim=1,
        )

        context_observed = past_observed_values[
            :, -self.model.context_length + 1 :
        ]
        observed_values = torch.cat(
            (context_observed, future_observed_values), dim=1
        )

        if len(self.model.target_shape) == 0:
            loss_weights = observed_values
        else:
            loss_weights = observed_values.min(dim=-1, keepdim=False)

        control_loss = weighted_average(self.loss(control_distr, control), weights=loss_weights)
        target_loss = weighted_average(self.loss(distr, target), weights=loss_weights)

        return control_loss, target_loss

    def training_step(self, batch, batch_idx: int):
        """Execute training step"""
        train_control_loss, train_target_loss = self._compute_loss(batch)
        train_loss = self.control_loss_weight * train_control_loss + train_target_loss
        self.log(
            "train_control_loss",
            train_control_loss,
            on_epoch=True,
            on_step=False,
            prog_bar=True,
        )
        self.log(
            "train_target_loss",
            train_target_loss,
            on_epoch=True,
            on_step=False,
            prog_bar=True,
        )
        self.log(
            "train_loss",
            train_loss,
            on_epoch=True,
            on_step=False,
            prog_bar=True,
        )
        return train_loss

    def validation_step(self, batch, batch_idx: int):
        """Execute validation step"""
        val_control_loss, val_target_loss = self._compute_loss(batch)
        val_loss = self.control_loss_weight * val_control_loss + val_target_loss
        self.log(
            "val_control_loss",
            val_control_loss,
            on_epoch=True,
            on_step=False,
            prog_bar=True,
        )
        self.log(
            "val_target_loss",
            val_target_loss,
            on_epoch=True,
            on_step=False,
            prog_bar=True,
        )
        self.log(
            "val_loss",
            val_loss,
            on_epoch=True,
            on_step=False,
            prog_bar=True,
        )
        return val_loss

    def configure_optimizers(self):
        """Returns the optimizer to use"""
        return torch.optim.Adam(
            self.model.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )
