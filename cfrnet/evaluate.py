import torch

from model import CfrNet
from config import Config
from utils import error_ATE, error_PEHE, error_ATT, R_pol, data_to_device


def evaluate(net: CfrNet, train_set, test_set, dataset):
    config = Config()
    results = {}
    with torch.no_grad():
        # Train test
        features_train = torch.from_numpy(train_set.x)
        treatment_type_train = torch.from_numpy(train_set.t)

        # Test set
        features_test = torch.from_numpy(test_set.x)
        treatment_type_test = torch.from_numpy(test_set.t)

        if config.use_gpu:
            features_train, treatment_type_train = data_to_device((features_train, treatment_type_train), config.device)

            features_test, treatment_type_test = data_to_device((features_test, treatment_type_test), config.device)

        y_h1_train, y_h0_train = net.predict(features_train)
        y_h1_test, y_h0_test = net.predict(features_test)
        if dataset == "ihdp":
            ate_error_train = error_ATE(y_h1_train.cpu(), y_h0_train.cpu(), test_set)
            pehe_error_train = error_PEHE(y_h1_train.cpu(), y_h0_train.cpu(), test_set)

            ate_error_test = error_ATE(y_h1_test.cpu(), y_h0_test.cpu(), test_set)
            pehe_error_test = error_PEHE(y_h1_test.cpu(), y_h0_test.cpu(), test_set)

            results["ate_error_train"] = ate_error_train
            results["pehe_error_train"] = pehe_error_train

            results["ate_error_test"] = ate_error_test
            results["pehe_error_test"] = pehe_error_test
        elif dataset == "jobs":
            # Train losses
            att_error_train = error_ATT(y_h1_train, y_h0_train, train_set)
            r_pol_train = R_pol(y_h1_train.cpu(), y_h0_train.cpu(), train_set)
            results["att_error_train"] = att_error_train
            results["r_pol_train"] = r_pol_train

            # Test losses
            att_error_test = error_ATT(y_h1_test, y_h0_test, test_set)
            r_pol_test = R_pol(y_h1_test.cpu(), y_h0_test.cpu(), test_set)
            results["att_error_test"] = att_error_test
            results["r_pol_test"] = r_pol_test

    return results
