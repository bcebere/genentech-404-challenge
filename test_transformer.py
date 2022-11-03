import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from hyperimpute.utils.serialization import load_model_from_file, save_model_to_file
from sklearn.preprocessing import MinMaxScaler
from baseline_imputation import prepare_consts, prepare_age

workspace = Path("workspace")
results_dir = Path("results")
data_dir = Path("data")

workspace.mkdir(parents=True, exist_ok=True)

warnings.filterwarnings("ignore")

cat_limit = 10
n_seeds = 5

version = "take8"
changelog = f"hyperlatent_transformer"


def dataframe_hash(df: pd.DataFrame) -> str:
    cols = sorted(list(df.columns))
    return str(abs(pd.util.hash_pandas_object(df[cols].fillna(0)).sum()))


def augment_base_dataset(df):
    df = df.sort_values(["RID_HASH", "VISCODE"])

    return df


dev_set = pd.read_csv(data_dir / "dev_set.csv")
dev_set = augment_base_dataset(dev_set)

scaled_cols = [
    "AGE",
    "PTEDUCAT",
    "MMSE",
    "ADAS13",
    "Ventricles",
    "Hippocampus",
    "WholeBrain",
    "Entorhinal",
    "Fusiform",
    "MidTemp",
]

scaler = MinMaxScaler().fit(dev_set[scaled_cols])
dev_set[scaled_cols] = scaler.transform(dev_set[scaled_cols])

dev_set

static_features = ["RID_HASH", "PTGENDER_num", "PTEDUCAT", "APOE4"]
temporal_features = [
    "RID_HASH",
    "VISCODE",
    "DX_num",
    "CDRSB",
    "MMSE",
    "ADAS13",
    "Ventricles",
    "Hippocampus",
    "WholeBrain",
    "Entorhinal",
    "Fusiform",
    "MidTemp",
]  #

dev_set_static = dev_set.sort_values(["RID_HASH", "VISCODE"]).drop_duplicates(
    "RID_HASH"
)[static_features]
dev_set_temporal = dev_set.sort_values(["RID_HASH", "VISCODE"])[temporal_features]

dev_set_static

dev_1 = pd.read_csv(data_dir / "dev_1.csv")
dev_1 = augment_base_dataset(dev_1)
dev_1[scaled_cols] = scaler.transform(dev_1[scaled_cols])

dev_1

dev_2 = pd.read_csv(data_dir / "dev_2.csv")
dev_2 = augment_base_dataset(dev_2)
dev_2[scaled_cols] = scaler.transform(dev_2[scaled_cols])

dev_2

test_A = pd.read_csv(data_dir / "test_A.csv")
test_A = augment_base_dataset(test_A)
test_A[scaled_cols] = scaler.transform(test_A[scaled_cols])

test_A_gt = pd.read_csv(data_dir / "test_A_gt.csv")
test_A_gt = augment_base_dataset(test_A_gt)
test_A_gt[scaled_cols] = scaler.transform(test_A_gt[scaled_cols])

assert (test_A["VISCODE"] == test_A_gt["VISCODE"]).all()

test_A_gt

test_B = pd.read_csv(data_dir / "test_B.csv")
test_B = augment_base_dataset(test_B)
test_B[scaled_cols] = scaler.transform(test_B[scaled_cols])

test_B_gt = pd.read_csv(data_dir / "test_B_gt.csv")
test_B_gt = augment_base_dataset(test_B_gt)
test_B_gt[scaled_cols] = scaler.transform(test_B_gt[scaled_cols])

assert (test_B["VISCODE"] == test_B_gt["VISCODE"]).all()

test_AB_input = pd.concat([test_A, test_B], ignore_index=True)
test_AB_output = pd.concat([test_A_gt, test_B_gt], ignore_index=True)

test_AB_output

from sklearn.preprocessing import LabelEncoder


def mask_columns_map(s: str):
    return f"masked_{s}"


def generate_testcase(ref_df, out_df, target_column: str, cat_thresh: int = cat_limit):
    assert len(ref_df) == len(out_df)
    ref_df = ref_df.sort_values(["RID_HASH", "VISCODE"]).reset_index(drop=True)
    out_df = out_df.sort_values(["RID_HASH", "VISCODE"]).reset_index(drop=True)

    assert (ref_df["RID_HASH"].values == out_df["RID_HASH"].values).all()
    assert (ref_df["VISCODE"].values == out_df["VISCODE"].values).all()

    mask = (
        ref_df.isna()
        .astype(int)
        .drop(columns=["RID_HASH", "VISCODE"])
        .rename(mask_columns_map, axis="columns")
    ).reset_index(drop=True)

    target_mask = (ref_df.isna().astype(int)).reset_index(drop=True)
    target_mask = target_mask[["RID_HASH", "VISCODE", target_column]]
    target_mask[["RID_HASH", "VISCODE"]] = ref_df[["RID_HASH", "VISCODE"]]

    ref_df = ref_df.fillna(-1)
    test_input = pd.concat([ref_df, mask], axis=1).reset_index(drop=True)
    test_output = out_df[["RID_HASH", "VISCODE", target_column]].reset_index(drop=True)

    if len(dev_set[target_column].unique()) < cat_thresh:
        encoding_data = pd.concat([dev_set, out_df, test_AB_output], ignore_index = True)
        encoder = LabelEncoder().fit(encoding_data[[target_column]])

        test_output[target_column] = encoder.transform(out_df[[target_column]])


    return test_input, test_output, target_mask


def prepare_dataset(target_column: str, cat_thresh: int = cat_limit):
    df_input_1, df_output_1, target_mask_1 = generate_testcase(
        dev_1, dev_set, target_column, cat_thresh=cat_thresh
    )
    df_input_2, df_output_2, target_mask_2 = generate_testcase(
        dev_2, dev_set, target_column, cat_thresh=cat_thresh
    )

    return (
        [df_input_1, df_output_1, target_mask_1],
        [df_input_2, df_output_2, target_mask_2],
    )


def prepare_test_dataset(target_column: str, cat_thresh: int = cat_limit):
    return generate_testcase(
        test_AB_input, test_AB_output, target_column, cat_thresh=cat_thresh
    )


from ts_imputer_v2 import TimeSeriesImputerTemporal
from sklearn.model_selection import train_test_split


def train_inputer_for_column(target_column: str, n_units_hidden: int = 150):
    bkp_file = workspace / f"dedicated_imputer_col_{target_column}_{n_units_hidden}.bkp"

    # if bkp_file.exists():
    #    return load_model_from_file(bkp_file)

    cat_thresh = 30
    testcases = prepare_dataset(target_column=target_column, cat_thresh=cat_thresh)
    test_in, test_output, test_target_mask = prepare_test_dataset(
        target_column=target_column, cat_thresh=cat_thresh
    )

    if len(dev_set[target_column].unique()) < cat_thresh:
        n_units_out = len(dev_set[target_column].unique())
        task_type = "classification"
    else:
        n_units_out = 1
        task_type = "regression"

    imputer = TimeSeriesImputerTemporal(
        task_type=task_type,
        n_units_in=testcases[0][0].shape[-1] - 1,  # DROP RID_HASH
        n_units_out=n_units_out,  # DROP RID_HASH and VISCODE
        nonlin="relu",
        dropout=0.05,
        # nonlin_out = activation_layout,
        n_layers_hidden=2,
        n_units_hidden=n_units_hidden,
        n_iter=10000,
        residual=False,
    )

    for repeat in range(3):
        for train_input, train_output, train_target_mask in testcases:
            imputer.fit(
                train_input,
                train_output,
                train_target_mask,
                test_in,
                test_output,
                test_target_mask,
            )
    save_model_to_file(bkp_file, imputer)

    return imputer


train_inputer_for_column(target_column = "WholeBrain")
#train_inputer_for_column(target_column="DX_num")
