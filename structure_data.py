import json
from typing import Literal, Union

import pandas as pd


def read_eqls() -> pd.DataFrame:
    return pd.read_csv("raw_data/csv/eqls_2007and2011.csv")


def read_column_dictionary() -> dict:
    with open("data/dictionary.json", "r") as file:
        return json.load(file)


def choose_columns() -> pd.DataFrame:
    eqls_data = read_eqls()
    column_dictionary = read_column_dictionary()

    treatment_column = None
    for column, info in column_dictionary.items():
        if info["causal_type"] == "treatment":
            treatment_column = column
            break

    outcome_column = None
    for column, info in column_dictionary.items():
        if info["causal_type"] == "outcome":
            outcome_column = column
            break

    if not treatment_column or not outcome_column:
        raise ValueError("Causal columns must include both treatment and outcome.")

    confounder_columns = []
    for column, info in column_dictionary.items():
        if info["causal_type"] == "confounder":
            confounder_columns.append(column)
    if not confounder_columns:
        raise ValueError("Causal columns must include at least one confounder.")

    print(f"[+] Treatment column: {treatment_column}")
    print(f"[+] Outcome column: {outcome_column}")
    print(f"[+] Confounder columns: {confounder_columns}")

    df = eqls_data[[treatment_column, outcome_column] + confounder_columns]
    df.rename(
        columns={treatment_column: "treatment", outcome_column: "outcome"}, inplace=True
    )
    return df


def preprocess_data(
    df: pd.DataFrame,
    na_threshold: float = 0.5,
    impute_strategy: Literal["drop", "mean", "median"] = "drop",
    treatment_dichotomize_value: Union[float, Literal["median"]] = "median",
) -> pd.DataFrame:

    # Handle missing values
    ## Drop all columns with more than na_threshold proportion of missing values
    df = df.dropna(axis=1, thresh=int(na_threshold * len(df)))

    ## Apply strategy for remaining missing values
    if impute_strategy == "drop":
        df = df.dropna(axis=0, how="any")
    elif impute_strategy == "mean":
        df = df.fillna(df.mean())
    elif impute_strategy == "median":
        df = df.fillna(df.median())

    # Dichotomize treatment variable
    if treatment_dichotomize_value == "median":
        treatment_threshold = df["treatment"].median()
    else:
        treatment_threshold: float = treatment_dichotomize_value
    df["treatment"] = (df["treatment"] > treatment_threshold).astype(int)

    return df


if __name__ == "__main__":
    df = choose_columns()
    df = preprocess_data(
        df,
        na_threshold=0.5,
        impute_strategy="drop",
        treatment_dichotomize_value="median",
    )
    df.to_csv("data/eqls_processed.csv", index=False)
