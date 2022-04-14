import pandas as pd
import torch as th


def maximum_mean_discrepancy_loss(X_treat, X_control):
    """Calculate Maximum Mean Discrepancy loss."""
    return 2 * th.norm(X_treat.mean(axis=0) - X_control.mean(axis=0))

if __name__ == "__main__":
    # For illustration purposes
    path = 'Data/ihdp10/csv/ihdp_npci_1.csv'
    feature_columns = [f"x{i}" for i in range(1, 26)]
    columns = ["treatment", "y_factual", "y_cfactual", "mu0", "mu1"] + feature_columns
    df = pd.read_csv(path, header=None)
    df.columns = columns

    treatment = df["treatment"].values.astype(bool)
    X_treat = th.from_numpy(df.loc[treatment, feature_columns].values)
    X_control = th.from_numpy(df.loc[~treatment, feature_columns].values)

    mmd = maximum_mean_discrepancy_loss(X_treat, X_control)

    print(f"{mmd}")
