# VISCODE 6 * x -> AGE 0.5 * x

const_by_patient = ["PTGENDER_num", "PTEDUCAT", "APOE4"]


def prepare_consts(test_data):
    test_data = test_data.copy()

    test_data = test_data.sort_values(["RID_HASH", "VISCODE"])

    for item in test_data.groupby("RID_HASH"):
        local = item[1]

        # fill consts
        for col in const_by_patient:
            if len(local[col].unique()) == 1:
                continue
            rid = local["RID_HASH"].unique()[0]

            val = local[col][~local[col].isna()].unique()[0]
            local[col] = local[col].fillna(val)
            test_data.loc[test_data["RID_HASH"] == rid, col] = test_data[
                test_data["RID_HASH"] == rid
            ][col].fillna(val)
            assert len(local[col].unique()) == 1, col

    return test_data


def prepare_age(test_data, scaler, scaled_cols):
    test_data = test_data.copy()
    test_data = test_data.sort_values(["RID_HASH", "VISCODE"])
    test_data[scaled_cols] = scaler.inverse_transform(test_data[scaled_cols])

    col = "AGE"

    for rid in test_data["RID_HASH"].unique():
        local = test_data[test_data["RID_HASH"] == rid]

        # fill age
        ages = local["AGE"]
        if ages.isna().sum() == 0:
            continue

        if ages.isna().sum() == len(ages):
            continue

        # forward impute age
        prev_viscode = 0
        prev_age = 0
        for idx, row in local.iterrows():
            current_viscode = row["VISCODE"]
            local_idx = (test_data["VISCODE"] == current_viscode) & (
                test_data["RID_HASH"] == rid
            )
            if prev_age > 0 and prev_age == prev_age:
                pred_age = (current_viscode - prev_viscode) / 6 * 0.5 + prev_age
            else:
                pred_age = row[col]

            if pred_age == pred_age:
                # print("forward imputed", pred_age, current_viscode)
                test_data.loc[local_idx, col] = test_data.loc[local_idx][col].fillna(
                    pred_age
                )

            prev_viscode = row["VISCODE"]
            prev_age = pred_age

        # reverse impute age
        prev_viscode = 0
        prev_age = 0
        for idx, row in local.iloc[::-1].iterrows():
            current_viscode = row["VISCODE"]
            local_idx = (test_data["VISCODE"] == current_viscode) & (
                test_data["RID_HASH"] == rid
            )

            if prev_age > 0 and prev_age == prev_age:
                pred_age = prev_age - (prev_viscode - current_viscode) / 6 * 0.5
            else:
                pred_age = row[col]

            if pred_age == pred_age:
                # print("reversed imputed", pred_age, current_viscode)
                test_data.loc[local_idx, col] = test_data.loc[local_idx][col].fillna(
                    pred_age
                )

            prev_viscode = row["VISCODE"]
            prev_age = pred_age

        # print(test_data[(test_data["RID_HASH"] == rid)][["VISCODE", "AGE"]])

    test_data[scaled_cols] = scaler.transform(test_data[scaled_cols])

    return test_data
