import random
from typing import Any, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from pydantic import validate_arguments
from torch import nn
from transformer import Transformer

from mlp import MLP

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def enable_reproducible_results(seed: int = 0) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)


class TimeSeriesImputerTemporal(nn.Module):
    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def __init__(
        self,
        n_units_in: int,
        n_units_out: int,
        n_units_hidden: int = 100,
        n_layers_hidden: int = 2,
        n_iter: int = 2500,
        n_iter_print: int = 100,
        batch_size: int = 100,
        lr: float = 1e-3,
        weight_decay: float = 1e-3,
        device: Any = DEVICE,
        nonlin_out: Optional[List[Tuple[str, int]]] = None,
        dropout: float = 0,
        nonlin: Optional[str] = "relu",
        random_state: int = 0,
        clipping_value: int = 1,
        patience: int = 5,
        residual: bool = True,
    ) -> None:
        super(TimeSeriesImputerTemporal, self).__init__()

        enable_reproducible_results(random_state)

        self.n_iter = n_iter
        self.n_iter_print = n_iter_print
        self.batch_size = batch_size
        self.n_units_in = n_units_in
        self.n_units_hidden = n_units_hidden
        self.n_layers_hidden = n_layers_hidden
        self.device = device
        self.lr = lr
        self.clipping_value = clipping_value

        self.patience = patience
        self.random_state = random_state

        self.layer_latent = Transformer(
            n_units_in=n_units_in,
            n_units_out=n_units_hidden,
            n_units_hidden=n_units_hidden,
            d_ffn=n_units_hidden,
            n_layers_hidden=n_layers_hidden,
            dropout=dropout,
            activation=nonlin,
        )

        self.layer_out = MLP(
            task_type="regression",
            n_units_in=n_units_hidden + 1,  # latent + VISCODE
            n_units_out=n_units_out,
            n_layers_hidden=n_layers_hidden,
            n_units_hidden=n_units_hidden,
            dropout=dropout,
            nonlin=nonlin,
            device=device,
            nonlin_out=nonlin_out,
            residual=residual,
        )
        self.nonlin_out = nonlin_out

        self.optimizer = torch.optim.Adam(
            self.parameters(),
            lr=lr,
            weight_decay=weight_decay,
        )  # optimize all rnn parameters

    def forward_latent(
        self,
        data: torch.Tensor,
        horizons: torch.Tensor
    ) -> torch.Tensor:
        assert torch.isnan(data).sum() == 0
        return self.layer_latent(
            torch.swapaxes(data, 1, 2),
            horizons
        )  # bs x seq_len x feats -> bs x feats x seq_len

    def forward(
        self,
        data: torch.Tensor,
        viscode: torch.Tensor,
    ) -> torch.Tensor:
        assert torch.isnan(data).sum() == 0
        print(data.shape, viscode.shape)
        viscode = viscode.unsqueeze(-1)

        pred = self.forward_latent(data)

        pred = pred.unsqueeze(1)
        pred = torch.repeat_interleave(pred, viscode.shape[1], dim=1)

        pred_merged = torch.cat([viscode, pred], dim=2)

        return self.layer_out(pred_merged)

    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def predict(
        self,
        miss_data: pd.DataFrame,
    ) -> Any:
        self.eval()
        miss_data = miss_data.copy()

        miss_data = miss_data.sort_values(["RID_HASH", "VISCODE"])
        patient_ids = miss_data["RID_HASH"].unique()

        output = pd.DataFrame([], columns=self.output_cols + ["VISCODE", "RID_HASH"])

        for rid in patient_ids:
            patient_miss = miss_data[miss_data["RID_HASH"] == rid]
            viscode = patient_miss["VISCODE"].values
            viscode = np.expand_dims(viscode, axis=0)

            patient_miss = patient_miss.drop(columns=["RID_HASH"])
            patient_miss = np.expand_dims(patient_miss.values, axis=0)

            patient_miss_t = self._check_tensor(patient_miss).float()
            viscode_t = self._check_tensor(viscode).float()

            temporal_preds = self(patient_miss_t, viscode_t)
            temporal_preds = temporal_preds.detach().cpu().numpy().squeeze()

            if len(temporal_preds.shape) == 1:
                temporal_preds = np.expand_dims(temporal_preds, axis=0)

            temporal_preds = pd.DataFrame(
                temporal_preds, columns=self.output_cols_temporal
            )
            temporal_preds["VISCODE"] = viscode.squeeze()
            temporal_preds["RID_HASH"] = rid

            output = pd.concat([output, temporal_preds], ignore_index=True)

            assert output.isna().sum().sum() == 0

        output.index = miss_data.index
        output = output[["RID_HASH", "VISCODE"] + self.output_cols_temporal]

        return output

    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def predict_latent(
        self,
        miss_data: pd.DataFrame,
    ) -> Any:
        self.eval()
        miss_data = miss_data.copy()

        miss_data = miss_data.sort_values(["RID_HASH", "VISCODE"])
        patient_ids = miss_data["RID_HASH"].unique()

        out_cols = ["RID_HASH"] + list(range(self.n_units_hidden))
        output = pd.DataFrame([], columns=out_cols)

        for rid in patient_ids:
            patient_miss = miss_data[miss_data["RID_HASH"] == rid]

            patient_miss = patient_miss.drop(columns=["RID_HASH"])
            patient_miss = np.expand_dims(patient_miss.values, axis=0)

            patient_miss_t = self._check_tensor(patient_miss).float()

            preds = self.forward_latent(patient_miss_t).detach().cpu().numpy().squeeze()
            if len(preds.shape) == 1:
                preds = np.expand_dims(preds, axis=0)

            preds = pd.DataFrame(preds, columns=list(range(self.n_units_hidden)))
            preds["RID_HASH"] = rid

            output = pd.concat([output, preds], ignore_index=True)

            assert output.isna().sum().sum() == 0

        output = output[out_cols]

        return output

    def eval_loss(self, preds, gt, nonlin_out=None, debug=False):
        loss = torch.tensor(0).to(self.device).float()

        if preds.shape[-1] == 0:
            return loss

        if nonlin_out is None:
            loss = nn.HuberLoss()(preds, gt)
        else:
            split = 0

            sanity_size = 0
            for _, step in nonlin_out:
                sanity_size += step

            assert sanity_size == preds.shape[-1]
            assert sanity_size == gt.shape[-1]
            for activation, step in nonlin_out:
                if activation == "softmax":
                    loss_fn = nn.CrossEntropyLoss()
                else:
                    loss_fn = nn.HuberLoss()
                local_loss = loss_fn(
                    preds[..., split : split + step],
                    gt[..., split : split + step],
                )

                if debug:
                    print(activation, step, local_loss)
                loss += local_loss

                split += step

        return loss

    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def fit(
        self,
        train_miss_data: pd.DataFrame,
        train_real_data: pd.DataFrame,
        val_miss_data: pd.DataFrame,
        val_real_data: pd.DataFrame,
    ) -> Any:
        assert len(train_miss_data) == len(train_real_data)

        train_miss_data = train_miss_data.sort_values(
            ["RID_HASH", "VISCODE"]
        ).reset_index(drop=True)
        train_real_data = train_real_data.sort_values(
            ["RID_HASH", "VISCODE"]
        ).reset_index(drop=True)

        val_miss_data = val_miss_data.sort_values(["RID_HASH", "VISCODE"]).reset_index(
            drop=True
        )
        val_real_data = val_real_data.sort_values(["RID_HASH", "VISCODE"]).reset_index(
            drop=True
        )

        patient_ids = train_miss_data["RID_HASH"].unique()

        batches_by_size = {}
        for rid in patient_ids:
            patient_miss = train_miss_data[train_miss_data["RID_HASH"] == rid]

            patient_seq_len = len(patient_miss)
            if patient_seq_len not in batches_by_size:
                batches_by_size[patient_seq_len] = {
                    "input": [],
                    "viscode": [],
                    "gt": [],
                }

            patient_gt = train_real_data[train_real_data["RID_HASH"] == rid]

            viscode = patient_gt["VISCODE"].values

            viscode = np.expand_dims(viscode, axis=0)

            patient_miss = patient_miss.drop(columns=["RID_HASH"])
            patient_miss = np.expand_dims(patient_miss.values, axis=0)

            patient_gt = patient_gt.drop(columns=["RID_HASH", "VISCODE"])

            self.output_cols = list(patient_gt.columns)
            patient_gt = np.expand_dims(patient_gt.values, axis=0)

            patient_miss_t = self._check_tensor(patient_miss).float()
            patient_gt_t = self._check_tensor(patient_gt).float()
            viscode_t = self._check_tensor(viscode).float()

            assert viscode_t.shape[1] == patient_gt.shape[1]
            assert viscode_t.shape[1] > 0, patient_gt

            batches_by_size[patient_seq_len]["input"].append(patient_miss_t)
            batches_by_size[patient_seq_len]["viscode"].append(viscode_t)
            batches_by_size[patient_seq_len]["gt"].append(patient_gt_t)

        batches = []
        for seq_len in batches_by_size:
            seq_miss_t = torch.cat(batches_by_size[seq_len]["input"])
            seq_viscode_t = torch.cat(batches_by_size[seq_len]["viscode"])
            seq_gt_temporal_t = torch.cat(batches_by_size[seq_len]["gt"])

            batches.append((seq_miss_t, seq_gt_temporal_t, seq_viscode_t))

        patience = 0
        best_val_loss = 999

        for epoch in range(self.n_iter):
            self.train()
            losses = []

            for (
                patient_miss_t,
                patient_gt_temporal_t,
                viscode_t,
            ) in batches:
                self.optimizer.zero_grad()

                temporal_preds = self(patient_miss_t, viscode_t)

                assert temporal_preds.shape == patient_gt_temporal_t.shape

                loss = self.eval_loss(
                    temporal_preds,
                    patient_gt_temporal_t,
                    nonlin_out=self.nonlin_out_temporal,
                )

                assert not torch.isnan(loss)

                loss.backward()  # backpropagation, compute gradients

                if self.clipping_value > 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.parameters(), self.clipping_value
                    )
                self.optimizer.step()  # apply gradients
                losses.append(loss.item())

            if (epoch + 1) % self.n_iter_print == 0:
                self.eval()
                gt_cols = list(val_real_data.columns)

                with torch.no_grad():
                    val_preds = self.predict(val_miss_data)
                    val_preds = (
                        val_preds[gt_cols]
                        .drop(columns=["RID_HASH", "VISCODE"])
                        .values.astype(float)
                    )

                    val_gt = val_real_data.drop(
                        columns=["RID_HASH", "VISCODE"]
                    ).values.astype(float)

                    val_loss = self.eval_loss(
                        torch.from_numpy(val_preds),
                        torch.from_numpy(val_gt),
                        nonlin_out=self.nonlin_out,
                    )
                    val_loss = val_loss.item()

                if val_loss < best_val_loss:
                    patience = 0
                    best_val_loss = val_loss
                else:
                    patience += 1

                if patience > self.patience:
                    # print(f"   >>> Epoch {epoch} Early stopping...")
                    break

                print(
                    f"   >>> Epoch {epoch} train loss  = {np.mean(losses)} val loss {val_loss}",
                    flush=True,
                )

        self.eval()
        return self

    def _check_tensor(self, X: torch.Tensor) -> torch.Tensor:
        if isinstance(X, torch.Tensor):
            return X.to(self.device)
        else:
            return torch.from_numpy(np.asarray(X)).to(self.device)
