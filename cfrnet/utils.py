from typing import List, Iterable

import pandas as pd
import torch
import numpy as np


def error_ATT(pred_y_t, pred_y_c, test_set):
    t = test_set.t
    y = test_set.y
    e = test_set.e
    """
    Calculates the error on the average treatment effect.
    :param pred_y_c: predicted y value for control
    :param pred_y_t: predicted y value for treated
    :param t: indication if sample was treated or not
    :param y: outcome of samples
    :param e: indication if sample was randomized
    :return: error on ATT value
    """

    # True ATT value
    # t = 0 and e = 1 indicates sample from C intersect E
    ATT = np.mean(y[t > 0]) - np.mean(y[(1 - t + e) > 1])

    # Predicted individual treatment effect
    ITE_pred = pred_y_t - pred_y_c

    # Only consider prediction on samples that were treated?
    # Their implementation takes (t + e > 1)
    ATT_pred = ITE_pred[(t > 0)].mean()

    err_ATT = abs(ATT - ATT_pred).item()

    return err_ATT


def ite(test_set):
    """
    Calculate the true Individual Treatment Effect (ITE) for all elements in the test set
    """
    pass


def error_ATE(y_h1, y_h0, test_set):
    """Calculate the error in the average treatment effect (ATE)

    We first compute the actual ATE.
    For this we construct the treated and control set, with the same size as the test set.
    We use the ycf field of the dataset to do so.

    First we build the treated set by concatenating the y values of the patients that were treated and the counterfactual
        y values of the patients that were not treated
    Second we build the control set by concatenating the y values of the patients that were not treated and the counterfactual
        y values of the patients that were treated

    The true ATE is then the mean of the differences between the treated and control sets
    The predicted ATE is the mean of the differences between the output of the h1 network and the h0 network.

    The error is their absolute difference
    """
    actual_treated_y = test_set.y[test_set.t]
    cf_treated_y = test_set.ycf[~test_set.t]
    treated = np.concatenate((actual_treated_y, cf_treated_y))

    actual_control_y = test_set.y[~test_set.t]
    cf_control_y = test_set.ycf[test_set.t]
    control = np.concatenate((cf_control_y, actual_control_y))

    ate_actual = (treated - control).mean()
    ate_pred = (y_h1 - y_h0).mean()

    return torch.abs(ate_actual - ate_pred).item()


def error_PEHE(y_h1, y_h0, test_set):
    actual_treated_y = test_set.y[test_set.t]
    cf_treated_y = test_set.ycf[~test_set.t]
    treated = np.concatenate((actual_treated_y, cf_treated_y))

    actual_control_y = test_set.y[~test_set.t]
    cf_control_y = test_set.ycf[test_set.t]
    control = np.concatenate((cf_control_y, actual_control_y))

    return np.sqrt(np.square((treated - control) - (y_h1.numpy() - y_h0.numpy())).mean())


def R_pol(pred_y_t, pred_y_c, test_set):
    t = test_set.t
    y = test_set.y
    """
    Calculates the average value loss when treating with a policy based on ITE.
    :param prediction: tensor of size (n,2): predicted y value for treated and non-treated
    :param t: indication if sample was treated or not
    :param y: outcome of samples
    :param e: indication if sample was randomized
    :return: policy risk
    """
    # ITE based on predictions
    ITE_pred = pred_y_t - pred_y_c

    # Treat if predicted ITE > lambda. Table 1 takes lambda = 0.
    lam = 0
    policy = (ITE_pred > lam).numpy()

    # Expectations of Y_0 and Y_1 given policy and t
    avg_treat_value = (y[(policy == t) * (t > 0)]).sum() / len(y)
    avg_control_value = (y[(policy == t) * (t < 1)]).sum() / len(y)

    # Probability of treating
    p = policy.mean()

    # Estimation of the policy risk
    policy_risk = 1 - p * avg_treat_value - (1 - p) * avg_control_value

    return policy_risk.item()


def get_data_with_treatment_type(data, treatment):
    treatment = treatment.squeeze()
    treated = data[treatment == 1]
    control = data[treatment == 0]
    return treated, control


def get_computing_device(use_gpu=False):
    return torch.device("cuda:0" if use_gpu and torch.cuda.is_available() else "cpu")


def data_to_device(data: Iterable[torch.Tensor], device) -> Iterable[torch.Tensor]:
    return (x.to(device) for x in data)


def results_to_df(all_results):
    keys = list(all_results[0].keys())
    new_dict = {k: [] for k in keys}
    for res in all_results:
        for k in keys:
            new_dict[k].append(res[k])

    return pd.DataFrame(new_dict)
