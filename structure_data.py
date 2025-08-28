import json
from typing import Literal, Union

import pandas as pd


def read_eqls() -> pd.DataFrame:
    return pd.read_csv("raw_data/csv/eqls_2007and2011.csv")


def read_column_dictionary() -> dict:
    with open("data/dictionary.json", "r") as file:
        return json.load(file)


def choose_columns(verbose=False) -> pd.DataFrame:
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

    covariate_columns = []
    for column, info in column_dictionary.items():
        if info["causal_type"] != "treatment" and info["causal_type"] != "outcome":
            covariate_columns.append(column)
    if not covariate_columns:
        raise ValueError("Causal columns must include at least one covariate.")

    if verbose:
        print(f"[+] Treatment column: {treatment_column}")
        print(f"[+] Outcome column: {outcome_column}")
        print(f"[+] Covariate columns: {covariate_columns}")
        for column in covariate_columns:
            na_count = eqls_data[column].isna().sum()
            na_percentage = na_count / len(eqls_data) * 100
            print(f"[+] {column}: {na_count} NA values ({na_percentage:.2f}%)")

    df = eqls_data[[treatment_column, outcome_column] + covariate_columns].copy()
    return df


def preprocess_data(
    df: pd.DataFrame,
    na_threshold: float = 0.5,
    impute_strategy: Literal["drop", "mean", "median", "keep"] = "drop",
    treatment_dichotomize_value: Union[float, Literal["median"]] = "median",
    treatment_column: str = "Y11_Q57",
    backdoor_variables: list[str] = None
) -> pd.DataFrame:

    # Handle missing values
    ## Drop all columns with more than na_threshold proportion of missing values
    df = df.dropna(axis=1, thresh=int(na_threshold * len(df)))

    ## Fix working hours bug
    df.loc[(df['Y11_EmploymentStatus'] != 1) & (df['Y11_Q7'].isna()), 'Y11_Q7'] = 0

    ## Apply strategy for remaining missing values
    if impute_strategy == "drop":
        if backdoor_variables:
            cols = [c for c in backdoor_variables if c in df.columns]
            if cols:
                df = df.dropna(axis=0, subset=cols, how="any")
            else:
                df = df.dropna(axis=0, how="any")
        else:
            df = df.dropna(axis=0, how="any")
    elif impute_strategy == "mean":
        df = df.fillna(df.mean(numeric_only=True))
    elif impute_strategy == "median":
        df = df.fillna(df.median(numeric_only=True))
    else:
        pass

    # Dichotomize treatment variable
    if treatment_dichotomize_value == "median":
        treatment_threshold = df[treatment_column].median()
    else:
        treatment_threshold: float = treatment_dichotomize_value
    df[treatment_column] = (df[treatment_column] > treatment_threshold).astype(int)
    return df


if __name__ == "__main__":
    df = choose_columns()
    df = preprocess_data(
        df,
        na_threshold=0.5,
        impute_strategy="keep",
        treatment_dichotomize_value="median",
    )
    df.to_csv("data/eqls_processed_dont_drop.csv", index=False)