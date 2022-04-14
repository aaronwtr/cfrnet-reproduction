import numpy as np
import pandas as pd

from torch.utils.data import Dataset

def open_jobs_data(dataset, datatype):
    """
    This function can open the jobs test, and train data. Note that there are 10 separate datasets and we need to
    explicitly specify which one we want to open.
    :param dataset: Integer specifying which dataset we want to open. Ranges from 0 to 9 for the jobs dataset.
    :param datatype: String specifying whether we want to open the test or train data.
    :return: Dataframe containing either the test or train data. x_i indicates a covariate, i.e. age, education, etc.
    t indicates treatment status. yf indicates the factual outcome and e is the indicator for whether the sample is
    an original randomized sample or whether it is non-randomized.
    sample.
    """

    if not isinstance(dataset, int):
        raise TypeError('Dataset can be an integer ranging from 0 to 9')

    if not isinstance(datatype, str):
        raise TypeError('Datatype should be a string, specifically either "test" or "train"')

    fredjo_path = '../Data/jobs_data/fredjo/'

    x_enum = range(17)
    x_labels = [f'x_{i}' for i in x_enum]

    if datatype == 'test':
        with np.load(f'{fredjo_path}jobs_DW_bin.new.10.test.npz') as data:
            test_data_x = pd.DataFrame(data['x'][:, :, dataset])
            test_data_x.columns = x_labels
            test_data_t = pd.DataFrame(data['t'][:, dataset])
            test_data_t.columns = ['t']
            test_data_yf = pd.DataFrame(data['yf'][:, dataset])
            test_data_yf.columns = ['yf']
            test_data_e = pd.DataFrame(data['e'][:, dataset])
            test_data_e.columns = ['e']
            test_data = pd.concat([test_data_x, test_data_t, test_data_yf, test_data_e], axis=1)
            return test_data

    if datatype == 'train':
        with np.load(f'{fredjo_path}jobs_DW_bin.new.10.train.npz') as data:
            train_data_x = pd.DataFrame(data['x'][:, :, dataset])
            train_data_x.columns = x_labels
            train_data_t = pd.DataFrame(data['t'][:, dataset])
            train_data_t.columns = ['t']
            train_data_yf = pd.DataFrame(data['yf'][:, dataset])
            train_data_yf.columns = ['yf']
            train_data_e = pd.DataFrame(data['e'][:, dataset])
            train_data_e.columns = ['e']
            train_data = pd.concat([train_data_x, train_data_t, train_data_yf, train_data_e], axis=1)
            return train_data


def open_ihdp_data(dataset, datatype):
    """
    This function can open the IHDP test, and train data. Note that there are 100 separate datasets and we need to
    explicitly specify which one we want to open.
    :param dataset: Integer specifying which dataset we want to open. Ranges from 0 to 999 for the jobs dataset.
    :param datatype: String specifying whether we want to open the test or train data.
    :return: Dataframe containing either the test or train data. x_i indicates a covariate, i.e. age, education, etc. t
    indicates treatment status. yf indicates the factual outcome and ycf indicates counterfactual outcome. mu0 and mu1
    are the noiseless potential outcomes.
    """

    if not isinstance(dataset, int):
        raise TypeError('Dataset can be an integer ranging from 0 to 999')

    if not isinstance(datatype, str):
        raise TypeError('Datatype should be a string, specifically either "test" or "train"')

    ihdp_path = '../Data/ihdp1000/'

    x_enum = range(25)
    x_labels = [f'x_{i}' for i in x_enum]

    if datatype == 'test':
        with np.load(f'{ihdp_path}ihdp_npci_1-1000.test.npz') as data:
            test_data_x = pd.DataFrame(data['x'][:, :, dataset])
            test_data_x.columns = x_labels
            test_data_t = pd.DataFrame(data['t'][:, dataset])
            test_data_t.columns = ['t']
            test_data_yf = pd.DataFrame(data['yf'][:, dataset])
            test_data_yf.columns = ['yf']
            test_data_ycf = pd.DataFrame(data['ycf'][:, dataset])
            test_data_ycf.columns = ['ycf']
            test_data_mu0 = pd.DataFrame(data['mu0'][:, dataset])
            test_data_mu0.columns = ['mu0']
            test_data_mu1 = pd.DataFrame(data['mu1'][:, dataset])
            test_data_mu1.columns = ['mu1']
            test_data = pd.concat([test_data_x, test_data_t, test_data_yf, test_data_ycf, test_data_mu0, test_data_mu1],
                                  axis=1)
            return test_data

    if datatype == 'train':
        with np.load(f'{ihdp_path}ihdp_npci_1-1000.train.npz') as data:
            train_data_x = pd.DataFrame(data['x'][:, :, dataset])
            train_data_x.columns = x_labels
            train_data_t = pd.DataFrame(data['t'][:, dataset])
            train_data_t.columns = ['t']
            train_data_yf = pd.DataFrame(data['yf'][:, dataset])
            train_data_yf.columns = ['yf']
            train_data_ycf = pd.DataFrame(data['ycf'][:, dataset])
            train_data_ycf.columns = ['ycf']
            train_data_mu0 = pd.DataFrame(data['mu0'][:, dataset])
            train_data_mu0.columns = ['mu0']
            train_data_mu1 = pd.DataFrame(data['mu1'][:, dataset])
            train_data_mu1.columns = ['mu1']
            train_data = pd.concat([train_data_x, train_data_t, train_data_yf, train_data_ycf, train_data_mu0,
                                    train_data_mu1], axis=1)
            return train_data


class IHDP(Dataset):
    """
    IHDP Dataset
    # Used this as reference: https://www.youtube.com/watch?v=PXOzkkB5eH0
    """

    def __init__(self, dataset, datatype):
        """
        :param data: dataframe array
        """
        xty = open_ihdp_data(dataset, datatype)
        # Drop the mu0 and mu1 they are not needed for the model.

        xty.drop(['mu0', 'mu1'], axis=1, inplace=True)

        self.t = xty.loc[:, ["t"]].to_numpy(np.bool)
        self.y = xty.loc[:, ["yf"]].to_numpy(np.float32)  # Dont include the counter factual data 'ycf' we need to learn this.
        self.ycf = xty.loc[:, ["ycf"]].to_numpy(np.float32)
        self.x = xty.drop(["yf", "ycf", "t"], axis=1).to_numpy(np.float32)
        self.n_samples = xty.shape[0]

        # Weights
        self.u = self.t.mean()
        self.weights = (self.t / 2 * self.u) + ((1 - self.t) / (2 * (1 - self.u)))

    def __getitem__(self, index):
        item = (
            self.x[index],
            self.t[index],
            self.y[index],
            self.weights[index]
        )

        # Returning (features len(=25), treatment len(=1), label len(=1))
        return item


    def __len__(self):
        return self.n_samples


class Jobs(Dataset):
    """
    Jobs Dataset
    """

    def __init__(self, dataset, datatype):
        xty = open_jobs_data(dataset, datatype)

        self.t = xty.loc[:, ["t"]].to_numpy(np.bool)
        self.y = xty.loc[:, ["yf"]].to_numpy(np.float32)
        self.x = xty.drop(["yf", "t", "e"], axis=1).to_numpy(np.float32)
        self.e = xty.loc[:, ["e"]].to_numpy(np.bool)
        self.n_samples = xty.shape[0]

        # Weights
        self.u = self.t.mean()
        self.weights = (self.t / 2 * self.u) + ((1 - self.t) / (2 * (1 - self.u)))


    def __getitem__(self, index):
        item = (
            self.x[index],
            self.t[index],
            self.y[index],
            self.weights[index]
        )

        return item


    def __len__(self):
        return self.n_samples
