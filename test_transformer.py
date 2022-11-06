import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from hyperimpute.utils.serialization import load_model_from_file, save_model_to_file
from sklearn.preprocessing import MinMaxScaler

from baseline_imputation import prepare_age, prepare_consts

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


def augment_base_dataset(df, scaler, scaled_cols):
    df = df.sort_values(["RID_HASH", "VISCODE"]).reset_index(drop=True)
    df = prepare_consts(df)
    df = prepare_age(df, scaler, scaled_cols)

    return df

dev_set = pd.read_csv(data_dir / "dev_set.csv")

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

dev_set = augment_base_dataset(dev_set, scaler, scaled_cols)

dev_set

static_features = ["RID_HASH", "AGE", "PTGENDER_num", "PTEDUCAT", "APOE4"]
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

raw_dev_1 = pd.read_csv(data_dir / "dev_1.csv")
dev_1 = augment_base_dataset(raw_dev_1, scaler, scaled_cols)
dev_1[scaled_cols] = scaler.transform(dev_1[scaled_cols])

dev_1

raw_dev_2 = pd.read_csv(data_dir / "dev_2.csv")
dev_2 = augment_base_dataset(raw_dev_2, scaler, scaled_cols)
dev_2[scaled_cols] = scaler.transform(dev_2[scaled_cols])

dev_2

raw_test_A = pd.read_csv(data_dir / "test_A.csv")
test_A = augment_base_dataset(raw_test_A, scaler, scaled_cols)
test_A[scaled_cols] = scaler.transform(test_A[scaled_cols])

test_A_gt = pd.read_csv(data_dir / "test_A_gt.csv")
test_A_gt = augment_base_dataset(test_A_gt, scaler, scaled_cols)
test_A_gt[scaled_cols] = scaler.transform(test_A_gt[scaled_cols])

assert (test_A["VISCODE"] == test_A_gt["VISCODE"]).all()

test_A_gt

raw_test_B = pd.read_csv(data_dir / "test_B.csv")
test_B = augment_base_dataset(raw_test_B, scaler, scaled_cols)
test_B[scaled_cols] = scaler.transform(test_B[scaled_cols])

test_B_gt = pd.read_csv(data_dir / "test_B_gt.csv")
test_B_gt = augment_base_dataset(test_B_gt, scaler, scaled_cols)
test_B_gt[scaled_cols] = scaler.transform(test_B_gt[scaled_cols])

assert (test_B["VISCODE"] == test_B_gt["VISCODE"]).all()

test_B

test_AB_input = pd.concat([test_A, test_B], ignore_index=True)

test_AB_raw = pd.concat([raw_test_A, raw_test_B], ignore_index=True)
test_AB_raw[scaled_cols] = scaler.transform(test_AB_raw[scaled_cols])
test_AB_raw = test_AB_raw.sort_values(["RID_HASH", "VISCODE"]).reset_index(drop=True)

test_AB_output = pd.concat([test_A_gt, test_B_gt], ignore_index=True)
test_AB_output = test_AB_output.sort_values(["RID_HASH", "VISCODE"]).reset_index(
    drop=True
)

test_AB_output

def copy_missingness(ref_data):
    ref_data_ids = ref_data["RID_HASH"].unique()

    len_to_miss = {}
    for rid in ref_data_ids:
        local_A = ref_data[ref_data["RID_HASH"] == rid]
        # print(len(local_A), local_A.isna().sum().sum())

        local_len = len(local_A)
        if local_len not in len_to_miss:
            len_to_miss[local_len] = []
        for reps in range(4):
            len_to_miss[local_len].append(local_A.notna().reset_index(drop=True))

    out_data = pd.DataFrame([], columns=dev_set.columns)
    out_data_ids = dev_set["RID_HASH"].unique()
    for rid in out_data_ids:
        local_A = dev_set[dev_set["RID_HASH"] == rid].copy().reset_index(drop=True)
        local_len = len(local_A)

        if local_len in len_to_miss and len(len_to_miss[local_len]) > 0:
            target_mask = len_to_miss[local_len].pop(0)
            out_data = pd.concat([out_data, local_A[target_mask]], ignore_index=True)
        else:
            out_data = pd.concat([out_data, local_A], ignore_index=True)

    return out_data

dev_sim_A = copy_missingness(test_A)

dev_sim_A

dev_sim_B = copy_missingness(test_B)

dev_sim_B

from sklearn.preprocessing import LabelEncoder


def mask_columns_map(s: str):
    return f"masked_{s}"


def generate_covariates(ref_df):
    ref_df = ref_df.sort_values(["RID_HASH", "VISCODE"]).reset_index(drop=True)
    mask = (
        ref_df.isna()
        .astype(int)
        .drop(columns=["RID_HASH", "VISCODE"])
        .rename(mask_columns_map, axis="columns")
    ).reset_index(drop=True)
    ref_df = ref_df.fillna(-1)
    test_input = pd.concat([ref_df, mask], axis=1).reset_index(drop=True)

    return test_input


def generate_testcase(ref_df, out_df, target_column: str, cat_thresh: int = cat_limit):
    assert len(ref_df) == len(out_df)
    ref_df = ref_df.sort_values(["RID_HASH", "VISCODE"]).reset_index(drop=True)
    out_df = out_df.sort_values(["RID_HASH", "VISCODE"]).reset_index(drop=True)

    assert (ref_df["RID_HASH"].values == out_df["RID_HASH"].values).all()
    assert (ref_df["VISCODE"].values == out_df["VISCODE"].values).all()

    test_input = generate_covariates(ref_df)

    target_mask = (ref_df.isna().astype(int)).reset_index(drop=True)
    target_mask = target_mask[["RID_HASH", "VISCODE", target_column]]
    target_mask[["RID_HASH", "VISCODE"]] = ref_df[["RID_HASH", "VISCODE"]]

    test_output = out_df[["RID_HASH", "VISCODE", target_column]].reset_index(drop=True)

    n_units_out = 1
    if len(dev_set[target_column].unique()) < cat_thresh:
        encoding_data = pd.concat([dev_set, out_df, test_AB_output], ignore_index=True)
        n_units_out = len(encoding_data[target_column].unique())

        encoder = LabelEncoder().fit(encoding_data[[target_column]])

        test_output[target_column] = encoder.transform(out_df[[target_column]])

    return test_input, test_output, target_mask, n_units_out


def prepare_dataset(target_column: str, cat_thresh: int = cat_limit):
    df_input_1, df_output_1, target_mask_1, n_units_out = generate_testcase(
        dev_1, dev_set, target_column, cat_thresh=cat_thresh
    )
    df_input_2, df_output_2, target_mask_2, _ = generate_testcase(
        dev_2, dev_set, target_column, cat_thresh=cat_thresh
    )
    df_input_sim_A, df_output_sim_A, target_mask_sim_A, _ = generate_testcase(
        dev_sim_A, dev_set, target_column, cat_thresh=cat_thresh
    )
    df_input_sim_B, df_output_sim_B, target_mask_sim_B, _ = generate_testcase(
        dev_sim_B, dev_set, target_column, cat_thresh=cat_thresh
    )

    return (
        [df_input_1, df_output_1, target_mask_1],
        [df_input_2, df_output_2, target_mask_2],
        [df_input_sim_A, df_output_sim_A, target_mask_sim_A],
        [df_input_sim_B, df_output_sim_B, target_mask_sim_B],
    ), n_units_out


def prepare_test_dataset(target_column: str, cat_thresh: int = cat_limit):
    test_input, test_output, test_mask, _ = generate_testcase(
        test_AB_input, test_AB_output, target_column, cat_thresh=cat_thresh
    )
    return test_input, test_output, test_mask

from sklearn.model_selection import train_test_split

from ts_imputer_v2 import TimeSeriesImputerTemporal


def get_imputer_for_column(target_column: str, n_units_hidden: int = 50):
    bkp_file = workspace / f"dedicated_imputer_col_{target_column}_{n_units_hidden}.bkp"

    print("Training imputer for", bkp_file)
    if bkp_file.exists():
        return load_model_from_file(bkp_file)

    cat_thresh = 30
    testcases, n_units_out = prepare_dataset(
        target_column=target_column, cat_thresh=cat_thresh
    )
    test_in, test_output, test_target_mask = prepare_test_dataset(
        target_column=target_column, cat_thresh=cat_thresh
    )

    if len(dev_set[target_column].unique()) < cat_thresh:
        task_type = "classification"
    else:
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
        patience = 5,
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

n_units_hidden = 50

for _target_col in reversed([
    "PTEDUCAT",
    "DX_num",
    "ADAS13",
    "Ventricles",
]):
    get_imputer_for_column(target_column=_target_col, n_units_hidden = n_units_hidden)


