import random
from typing import Any, Callable, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from pydantic import validate_arguments
from torch import nn
from tsai.models.ResCNN import ResCNN
from tsai.models.TransformerModel import TransformerModel
from tsai.models.XceptionTime import XceptionTime

from mlp import MLP

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


modes = [
    "LSTM",
    "GRU",
    "RNN",
    "Transformer",
    "XceptionTime",
    "ResCNN",
]


def enable_reproducible_results(seed: int = 0) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)


class TimeSeriesImputer(nn.Module):
    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def __init__(
        self,
        n_units_in: int,
        n_units_out_static: int,
        n_units_out_temporal: int,
        n_units_hidden: int = 100,
        n_layers_hidden: int = 2,
        n_iter: int = 2500,
        mode: str = "RNN",
        n_iter_print: int = 100,
        batch_size: int = 100,
        lr: float = 1e-3,
        weight_decay: float = 1e-3,
        device: Any = DEVICE,
        nonlin_out_static: Optional[List[Tuple[str, int]]] = None,
        nonlin_out_temporal: Optional[List[Tuple[str, int]]] = None,
        dropout: float = 0,
        nonlin: Optional[str] = "leaky_relu",
        random_state: int = 0,
        clipping_value: int = 1,
        patience: int = 5,
        residual: bool = True,
    ) -> None:
        super(TimeSeriesImputer, self).__init__()

        enable_reproducible_results(random_state)

        assert mode in modes, f"Unsupported mode {mode}. Available: {modes}"

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

        self.mode = mode

        self.layer_latent = TimeSeriesLayer(
            n_units_in=n_units_in,
            n_units_out=n_units_hidden,
            n_units_hidden=n_units_hidden,
            n_layers_hidden=n_layers_hidden,
            mode=mode,
            device=device,
            dropout=dropout,
            nonlin=nonlin,
            random_state=random_state,
        )

        self.layer_out_static = MLP(
            task_type="regression",
            n_units_in=n_units_hidden,  # latent
            n_units_out=n_units_out_static,
            n_layers_hidden=n_layers_hidden,
            n_units_hidden=n_units_hidden,
            dropout=dropout,
            nonlin=nonlin,
            device=device,
            nonlin_out=nonlin_out_static,
            residual=residual,
        )
        self.layer_out_temporal = MLP(
            task_type="regression",
            n_units_in=n_units_hidden + 1,  # latent + VISCODE
            n_units_out=n_units_out_temporal,
            n_layers_hidden=n_layers_hidden,
            n_units_hidden=n_units_hidden,
            dropout=dropout,
            nonlin=nonlin,
            device=device,
            nonlin_out=nonlin_out_temporal,
            residual=residual,
        )
        self.nonlin_out_static = nonlin_out_static
        self.nonlin_out_temporal = nonlin_out_temporal

        self.optimizer = torch.optim.Adam(
            self.parameters(),
            lr=lr,
            weight_decay=weight_decay,
        )  # optimize all rnn parameters

    def forward_latent(
        self,
        data: torch.Tensor,
    ) -> torch.Tensor:
        assert torch.isnan(data).sum() == 0
        return self.layer_latent(data)

    def forward(
        self,
        data: torch.Tensor,
        viscode: torch.Tensor,
    ) -> torch.Tensor:
        assert torch.isnan(data).sum() == 0
        viscode = viscode.unsqueeze(-1)

        pred = self.forward_latent(data)

        static_preds = self.layer_out_static(pred)

        pred = pred.unsqueeze(1)
        pred = torch.repeat_interleave(pred, viscode.shape[1], dim=1)

        pred_merged = torch.cat([viscode, pred], dim=2)

        temporal_preds = self.layer_out_temporal(pred_merged)

        return static_preds, temporal_preds

    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def predict(
        self,
        miss_data: pd.DataFrame,
    ) -> Any:
        self.eval()
        miss_data = miss_data.copy()

        miss_data = miss_data.sort_values(["RID_HASH", "VISCODE"])
        patient_ids = miss_data["RID_HASH"].unique()

        output_static = pd.DataFrame([], columns=self.output_cols_static + ["RID_HASH"])
        output_temporal = pd.DataFrame(
            [], columns=self.output_cols_temporal + ["VISCODE", "RID_HASH"]
        )

        for rid in patient_ids:
            patient_miss = miss_data[miss_data["RID_HASH"] == rid]
            viscode = patient_miss["VISCODE"].values
            viscode = np.expand_dims(viscode, axis=0)

            patient_miss = patient_miss.drop(columns=["RID_HASH"])
            patient_miss = np.expand_dims(patient_miss.values, axis=0)

            patient_miss_t = self._check_tensor(patient_miss).float()
            viscode_t = self._check_tensor(viscode).float()

            static_preds, temporal_preds = self(patient_miss_t, viscode_t)
            static_preds = static_preds.detach().cpu().numpy().squeeze()
            temporal_preds = temporal_preds.detach().cpu().numpy().squeeze()

            if len(temporal_preds.shape) == 1:
                temporal_preds = np.expand_dims(temporal_preds, axis=0)

            if len(static_preds.shape) == 1:
                static_preds = np.expand_dims(static_preds, axis=0)

            static_preds = pd.DataFrame(static_preds, columns=self.output_cols_static)
            static_preds["RID_HASH"] = rid

            temporal_preds = pd.DataFrame(
                temporal_preds, columns=self.output_cols_temporal
            )
            temporal_preds["VISCODE"] = viscode.squeeze()
            temporal_preds["RID_HASH"] = rid

            output_static = pd.concat([output_static, static_preds], ignore_index=True)
            output_temporal = pd.concat(
                [output_temporal, temporal_preds], ignore_index=True
            )

            assert output_static.isna().sum().sum() == 0
            assert output_temporal.isna().sum().sum() == 0

        output_temporal.index = miss_data.index
        output_temporal = output_temporal[
            ["RID_HASH", "VISCODE"] + self.output_cols_temporal
        ]
        output_static = output_static[["RID_HASH"] + self.output_cols_static]

        return output_static, output_temporal

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
        if nonlin_out is None:
            loss = nn.HuberLoss()(preds, gt)
        else:
            loss = 0
            split = 0

            sanity_size = 0
            for _, step in nonlin_out:
                sanity_size += step

            assert sanity_size == preds.shape[-1]
            assert sanity_size == gt.shape[-1]

            for activation, step in nonlin_out:
                if activation == "softmax":
                    loss_fn = nn.CrossEntropyLoss()
                    factor = 1
                else:
                    loss_fn = nn.HuberLoss()
                    factor = 100
                local_loss = loss_fn(
                    preds[..., split : split + step],
                    gt[..., split : split + step],
                )
                if debug:
                    print(activation, step, factor * local_loss)
                loss += factor * local_loss

                split += step
        return loss

    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def fit(
        self,
        train_miss_data: pd.DataFrame,
        train_real_data_static: pd.DataFrame,
        train_real_data_temporal: pd.DataFrame,
        val_miss_data: pd.DataFrame,
        val_real_data_static: pd.DataFrame,
        val_real_data_temporal: pd.DataFrame,
    ) -> Any:
        assert len(train_miss_data) == len(train_real_data_temporal)

        train_miss_data = train_miss_data.sort_values(
            ["RID_HASH", "VISCODE"]
        ).reset_index(drop=True)
        train_real_data_temporal = train_real_data_temporal.sort_values(
            ["RID_HASH", "VISCODE"]
        ).reset_index(drop=True)
        train_real_data_static = train_real_data_static.sort_values(
            ["RID_HASH"]
        ).reset_index(drop=True)

        val_miss_data = val_miss_data.sort_values(["RID_HASH", "VISCODE"]).reset_index(
            drop=True
        )
        val_real_data_temporal = val_real_data_temporal.sort_values(
            ["RID_HASH", "VISCODE"]
        ).reset_index(drop=True)
        val_real_data_static = val_real_data_static.sort_values(
            ["RID_HASH"]
        ).reset_index(drop=True)
        val_real_data_static = val_real_data_static.drop_duplicates("RID_HASH")

        patient_ids = train_miss_data["RID_HASH"].unique()

        batches_by_size = {}
        for rid in patient_ids:
            patient_miss = train_miss_data[train_miss_data["RID_HASH"] == rid]

            patient_seq_len = len(patient_miss)
            if patient_seq_len not in batches_by_size:
                batches_by_size[patient_seq_len] = {
                    "input": [],
                    "viscode": [],
                    "gt_temporal": [],
                    "gt_static": [],
                }

            patient_gt_temporal = train_real_data_temporal[
                train_real_data_temporal["RID_HASH"] == rid
            ]
            patient_gt_static = train_real_data_static[
                train_real_data_static["RID_HASH"] == rid
            ]
            patient_gt_static = patient_gt_static.drop_duplicates("RID_HASH")

            viscode = patient_gt_temporal["VISCODE"].values
            viscode = np.expand_dims(viscode, axis=0)

            patient_miss = patient_miss.drop(columns=["RID_HASH"])
            patient_miss = np.expand_dims(patient_miss.values, axis=0)

            patient_gt_temporal = patient_gt_temporal.drop(
                columns=["RID_HASH", "VISCODE"]
            )
            patient_gt_static = patient_gt_static.drop(columns=["RID_HASH"])

            self.output_cols_temporal = list(patient_gt_temporal.columns)
            self.output_cols_static = list(patient_gt_static.columns)
            patient_gt_temporal = np.expand_dims(patient_gt_temporal.values, axis=0)

            patient_miss_t = self._check_tensor(patient_miss).float()
            patient_gt_temporal_t = self._check_tensor(patient_gt_temporal).float()
            patient_gt_static_t = self._check_tensor(patient_gt_static).float()
            viscode_t = self._check_tensor(viscode).float()

            assert viscode_t.shape[1] == patient_gt_temporal_t.shape[1]

            batches_by_size[patient_seq_len]["input"].append(patient_miss_t)
            batches_by_size[patient_seq_len]["viscode"].append(viscode_t)
            batches_by_size[patient_seq_len]["gt_temporal"].append(
                patient_gt_temporal_t
            )
            batches_by_size[patient_seq_len]["gt_static"].append(patient_gt_static_t)

        batches = []
        for seq_len in batches_by_size:
            seq_miss_t = torch.cat(batches_by_size[seq_len]["input"])
            seq_viscode_t = torch.cat(batches_by_size[seq_len]["viscode"])
            seq_gt_temporal_t = torch.cat(batches_by_size[seq_len]["gt_temporal"])
            seq_gt_static_t = torch.cat(batches_by_size[seq_len]["gt_static"])

            batches.append(
                (seq_miss_t, seq_gt_static_t, seq_gt_temporal_t, seq_viscode_t)
            )

        patience = 0
        best_val_loss = 999

        for epoch in range(self.n_iter):
            self.train()
            losses_static = []
            losses_temporal = []
            for (
                patient_miss_t,
                patient_gt_static_t,
                patient_gt_temporal_t,
                viscode_t,
            ) in batches:
                self.optimizer.zero_grad()

                static_preds, temporal_preds = self(patient_miss_t, viscode_t)

                assert static_preds.shape == patient_gt_static_t.shape
                assert temporal_preds.shape == patient_gt_temporal_t.shape

                loss_static = self.eval_loss(
                    static_preds, patient_gt_static_t, nonlin_out=self.nonlin_out_static
                )
                loss_temporal = self.eval_loss(
                    temporal_preds,
                    patient_gt_temporal_t,
                    nonlin_out=self.nonlin_out_temporal,
                )

                loss = loss_static + loss_temporal
                loss.backward()  # backpropagation, compute gradients

                if self.clipping_value > 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.parameters(), self.clipping_value
                    )
                self.optimizer.step()  # apply gradients
                losses_static.append(loss_static.item())
                losses_temporal.append(loss_temporal.item())

            if (epoch + 1) % self.n_iter_print == 0:
                self.eval()
                gt_cols_temporal = list(val_real_data_temporal.columns)
                gt_cols_static = list(val_real_data_static.columns)

                with torch.no_grad():
                    val_preds_static, val_preds_temporal = self.predict(val_miss_data)
                    val_preds_static = (
                        val_preds_static[gt_cols_static]
                        .drop(columns=["RID_HASH"])
                        .values.astype(float)
                    )
                    val_preds_temporal = (
                        val_preds_temporal[gt_cols_temporal]
                        .drop(columns=["RID_HASH", "VISCODE"])
                        .values.astype(float)
                    )

                    val_gt_static = val_real_data_static.drop(
                        columns=["RID_HASH"]
                    ).values.astype(float)
                    val_gt_temporal = val_real_data_temporal.drop(
                        columns=["RID_HASH", "VISCODE"]
                    ).values.astype(float)

                    val_loss_static = self.eval_loss(
                        torch.from_numpy(val_preds_static),
                        torch.from_numpy(val_gt_static),
                        nonlin_out=self.nonlin_out_static,
                    ).item()
                    val_loss_temporal = self.eval_loss(
                        torch.from_numpy(val_preds_temporal),
                        torch.from_numpy(val_gt_temporal),
                        nonlin_out=self.nonlin_out_temporal,
                    ).item()
                    val_loss = val_loss_static + val_loss_temporal

                if val_loss < best_val_loss:
                    patience = 0
                    best_val_loss = val_loss
                else:
                    patience += 1

                if patience > self.patience:
                    # print(f"   >>> Epoch {epoch} Early stopping...")
                    break

                print(
                    f"   >>> Epoch {epoch} train loss static = {np.mean(losses_static)}, temporal = {np.mean(losses_temporal)}. val loss static = {val_loss_static}, temporal = {val_loss_temporal}",
                    flush=True,
                )

        self.eval()
        return self

    def _check_tensor(self, X: torch.Tensor) -> torch.Tensor:
        if isinstance(X, torch.Tensor):
            return X.to(self.device)
        else:
            return torch.from_numpy(np.asarray(X)).to(self.device)


class TimeSeriesLayer(nn.Module):
    def __init__(
        self,
        n_units_in: int,
        n_units_out: int,
        n_units_hidden: int = 100,
        n_layers_hidden: int = 2,
        mode: str = "RNN",
        device: Any = DEVICE,
        dropout: float = 0,
        nonlin: Optional[str] = "relu",
        random_state: int = 0,
    ) -> None:
        super(TimeSeriesLayer, self).__init__()
        temporal_params = {
            "input_size": n_units_in,
            "hidden_size": n_units_hidden,
            "num_layers": n_layers_hidden,
            "dropout": 0 if n_layers_hidden == 1 else dropout,
            "batch_first": True,
        }
        temporal_models = {
            "RNN": nn.RNN,
            "LSTM": nn.LSTM,
            "GRU": nn.GRU,
        }

        if mode in ["RNN", "LSTM", "GRU"]:
            self.temporal_layer = temporal_models[mode](**temporal_params)
        elif mode == "XceptionTime":
            self.temporal_layer = XceptionTime(
                c_in=n_units_in,
                c_out=n_units_out,
            )
        elif mode == "ResCNN":
            self.temporal_layer = ResCNN(
                c_in=n_units_in,
                c_out=n_units_out,
            )
        elif mode == "Transformer":
            self.temporal_layer = TransformerModel(
                c_in=n_units_in,
                c_out=n_units_out,
                d_model=n_units_hidden,
                dropout=dropout,
                n_layers=n_layers_hidden,
            )
        else:
            raise RuntimeError(f"Unknown TS mode {mode}")

        self.n_layers_hidden = n_layers_hidden
        self.n_units_hidden = n_units_hidden

        if mode in ["RNN", "LSTM", "GRU"]:
            self.out_layer = MLP(
                task_type="regression",
                n_units_in=n_units_hidden,
                n_units_out=n_units_out,
                n_layers_hidden=1,
                n_units_hidden=n_units_hidden,
                dropout=dropout,
                nonlin=nonlin,
                device=device,
            )

        self.device = device
        self.mode = mode

        self.temporal_layer.to(device)

    def forward(self, temporal_data: torch.Tensor) -> torch.Tensor:
        if self.mode in ["RNN", "LSTM", "GRU"]:
            X_interm, _ = self.temporal_layer(temporal_data)
            X_interm = X_interm[:, -1:, :].reshape(-1, self.n_units_hidden)
            assert torch.isnan(X_interm).sum() == 0
            return self.out_layer(X_interm)
        else:
            X_interm = self.temporal_layer(torch.swapaxes(temporal_data, 1, 2))
            assert torch.isnan(X_interm).sum() == 0
            return X_interm
