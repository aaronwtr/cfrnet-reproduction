import torch
from utils import get_data_with_treatment_type


class LogisticRegression(torch.nn.Module):
    """
    Implement PyTorch logistic regression with OLS loss.

    Note that to obtain OLS1 we need to consider treatment data as our features. For OLS2 we need seperate regressors
    for each treatment. ? Don't quite know what this means yet.
    """

    def __init__(self, input_dim):
        super(LogisticRegression, self).__init__()
        self.linear = torch.nn.Linear(input_dim, 1)

    def forward(self, x):
        """
        Forward pass
        :param x: Input data
        :return: Sigmoidal mapping of linear transformation with learnable feature weights and bias as parameters
        sigmoid(y = x * w + b)
        """
        return torch.sigmoid(self.linear(x))


class LogisticRegressionNet():
    """Class that encapsulates OLS-1 and OLS-2 methods.
    Similar to what CFRNet is to the Net class in model.py
    """

    def __init__(self, input_dim, config):
        self.split_h = config.split_h
        if self.split_h:
            self.h1 = LogisticRegression(input_dim)
            self.h0 = LogisticRegression(input_dim)
            all_parameters = list(self.h1.parameters()) + list(self.h0.parameters())
        else:
            self.h = LogisticRegression(input_dim + 1)
            all_parameters = self.h.parameters()

        self.criterion = torch.nn.MSELoss()
        self.optim = torch.optim.Adam(all_parameters, lr=config.learning_rate)

    def calculate_loss(self, data, config):
        features, treatment_type, labels, weights = data

        if self.split_h:
            features_treated, features_control = get_data_with_treatment_type(features, treatment_type)
            treated_y, control_y = get_data_with_treatment_type(labels, treatment_type)
            y_pred_h1 = self.h1(features_treated)
            y_pred_h0 = self.h0(features_control)

            loss = self.criterion(y_pred_h1, treated_y)
            loss += self.criterion(y_pred_h0, control_y)
        else:
            features_with_treatment = torch.cat((features, treatment_type), dim=1)
            y_pred = self.h(features_with_treatment)
            loss = self.criterion(y_pred, labels)
        return loss

    def predict(self, features):
        """Predict treated and control y for the features"""
        if self.split_h:
            y_h1 = self.h1(features)
            y_h0 = self.h0(features)
        else:
            y_h1 = self.h(torch.cat((features, torch.ones(features.shape[0], 1)), dim=1))
            y_h0 = self.h(torch.cat((features, torch.zeros(features.shape[0], 1)), dim=1))
        return y_h1, y_h0
