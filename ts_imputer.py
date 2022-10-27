from typing import Any, Optional, List, Tuple, Callable
from torch import nn
import torch
import random
import numpy as np
import pandas as pd

from mlp import MLP
from pydantic import validate_arguments

from tsai.models.TCN import TCN
from tsai.models.TransformerModel import TransformerModel
from tsai.models.ResCNN import ResCNN
from tsai.models.XceptionTime import XceptionTime

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


modes = [
    "LSTM",
    "GRU",
    "RNN",
    "Transformer",
    "ResCNN",
    "TCN",
    "XceptionTime",
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
        n_units_out: int,
        n_units_hidden: int = 200,
        n_layers_hidden: int = 1,
        n_iter: int = 2500,
        mode: str = "RNN",
        n_iter_print: int = 10,
        batch_size: int = 100,
        lr: float = 1e-3,
        weight_decay: float = 1e-3,
        device: Any = DEVICE,
        nonlin_out: Optional[List[Tuple[str, int]]] = None,
        loss: Optional[Callable] = None,
        dropout: float = 0,
        nonlin: Optional[str] = "relu",
        random_state: int = 0,
        clipping_value: int = 1,
        patience: int = 20,
        train_ratio: float = 0.8,
        residual: bool = False,
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
        self.train_ratio = train_ratio
        self.random_state = random_state

        self.temporal_layer = TimeSeriesLayer(
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

        self.mode = mode

        self.out_layer = MLP(
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
        self.loss = nn.MSELoss()
        self.nonlin_out = nonlin_out

        self.optimizer = torch.optim.Adam(
            self.parameters(),
            lr=lr,
            weight_decay=weight_decay,
        )  # optimize all rnn parameters

    def forward(
        self,
        data: torch.Tensor,
        viscode: torch.Tensor,
    ) -> torch.Tensor:
        # x shape (batch, time_step, input_size)
        # r_out shape (batch, time_step, output_size)

        assert torch.isnan(data).sum() == 0
        viscode = viscode.squeeze()
        viscode = viscode.unsqueeze(1)

        pred = self.temporal_layer(data)
        pred = torch.repeat_interleave(pred, len(viscode), dim=0)
        pred_merged = torch.cat([viscode, pred], dim=1)

        return self.out_layer(pred_merged)

    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def fit(
        self,
        miss_data: pd.DataFrame,
        real_data: pd.DataFrame,
    ) -> Any:
        assert len(miss_data) == len(real_data)

        miss_data = miss_data.sort_values(["RID_HASH", "VISCODE"])
        real_data = real_data.sort_values(["RID_HASH", "VISCODE"])

        patient_ids = miss_data["RID_HASH"].unique()

        batches = []
        for rid in patient_ids:
            patient_miss = miss_data[miss_data["RID_HASH"] == rid]
            patient_gt = real_data[real_data["RID_HASH"] == rid]

            viscode = patient_gt["VISCODE"]
            patient_miss = patient_miss.drop(columns=["RID_HASH"])
            patient_miss = np.expand_dims(patient_miss.values, axis=0)

            patient_gt = patient_gt.drop(columns=["RID_HASH", "VISCODE"])
            patient_gt = np.expand_dims(patient_gt.values, axis=0)

            patient_miss_t = self._check_tensor(patient_miss).float()
            patient_gt_t = self._check_tensor(patient_gt).float()
            viscode_t = self._check_tensor(viscode).float()

            assert len(patient_miss_t) == len(patient_gt_t)
            assert len(viscode_t) == patient_gt_t.shape[1]

            batches.append((patient_miss_t, patient_gt_t, viscode_t))

        for epoch in range(self.n_iter):
            losses = []
            for patient_miss_t, patient_gt_t, viscode_t in batches:
                self.optimizer.zero_grad()

                preds = self(patient_miss_t, viscode_t)

                patient_gt_t = patient_gt_t.squeeze()
                if self.nonlin_out is None:
                    loss = nn.MSELoss()(preds, patient_gt_t)
                else:
                    loss = 0
                    split = 0
                    for activation, step in self.nonlin_out:
                        if activation == "softmax":
                            loss_fn = nn.CrossEntropyLoss()
                        else:
                            loss_fn = nn.MSELoss()
                        loss += loss_fn(
                            preds[:, split : split + step],
                            patient_gt_t[:, split : split + step],
                        )

                        split += step

                loss.backward()  # backpropagation, compute gradients

                if self.clipping_value > 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.parameters(), self.clipping_value
                    )
                self.optimizer.step()  # apply gradients
                losses.append(loss.item())
            print(f"Epoch {epoch} loss {np.mean(losses)}")
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
        elif mode == "TCN":
            self.temporal_layer = TCN(
                c_in=n_units_in,
                c_out=n_units_hidden,
                fc_dropout=dropout,
            )
        elif mode == "XceptionTime":
            self.temporal_layer = XceptionTime(
                c_in=n_units_in,
                c_out=n_units_hidden,
            )
        elif mode == "ResCNN":
            self.temporal_layer = ResCNN(
                c_in=n_units_in,
                c_out=n_units_hidden,
            )
        elif mode == "Transformer":
            self.temporal_layer = TransformerModel(
                c_in=n_units_in,
                c_out=n_units_hidden,
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
                n_units_in=n_units_hidden * n_layers_hidden,
                n_units_out=n_units_out,
                n_layers_hidden=1,
                n_units_hidden=n_units_hidden,
                dropout=dropout,
                nonlin=nonlin,
                device=device,
            )
        else:
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
            X_interm = X_interm[:, -self.n_layers_hidden :, :].reshape(
                -1, self.n_layers_hidden * self.n_units_hidden
            )
        else:
            X_interm = self.temporal_layer(torch.swapaxes(temporal_data, 1, 2))

        assert torch.isnan(X_interm).sum() == 0

        return self.out_layer(X_interm)
