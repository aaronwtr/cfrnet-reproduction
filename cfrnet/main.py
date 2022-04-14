import os.path
from datetime import datetime
from test import test
from time import time

from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import Config
from data_preprocessing import IHDP, Jobs
from evaluate import evaluate
from lr_model import LogisticRegressionNet
from model import CfrNet
from plot import plot_results
from train import train
from utils import results_to_df

def train_test_loop(net, train_set, test_set):
    config = Config()

    kwargs = {'num_workers': 2, 'pin_memory': True} if config.use_gpu else {}
    train_loader = DataLoader(train_set, batch_size=config.batch_size, shuffle=True, **kwargs)
    test_loader = DataLoader(test_set, batch_size=config.batch_size, shuffle=True, **kwargs)

    train_losses = []
    test_losses = []

    iter = range(config.num_epochs)
    if config.do_log_epochs:
        iter = tqdm(iter)

    for epoch in iter:
        train_loss = train(train_loader, net, config)
        train_losses.append(train_loss)

        test_loss = test(test_loader, net, config)
        test_losses.append(test_loss)
        if config.do_log_epochs and epoch % 10 == 0:
            print(f"Epoch {epoch}, train loss: {train_loss}, test loss: {test_loss}")

    return train_losses, test_losses


if __name__ == '__main__':
    # Jobs dataset, args: 24, 200, 200.
    # IHDP dataset args: 24, 200 200
    now = datetime.now()
    str_date_time = now.strftime("%d-%m-%Y_%H-%M-%S")
    config = Config()

    if not os.path.isdir(config.output_dir) and config.do_save:
        os.makedirs(config.output_dir)

    all_results = []
    n = config.n_iterations
    dataset = config.dataset
    model_name = config.model_name

    if model_name == "cfrnet":
        ipm_function = f"{config.ipm_function}_"
    else:
        ipm_function = ""
    name = f"{dataset}{n}_{model_name}_{ipm_function}{str_date_time}"
    print(f"Output name: {name}")
    path = os.path.join(config.output_dir, name)
    for i in range(n):
        print("Iteration:", i)
        if dataset == "ihdp":
            train_set = IHDP(i, "train")
            test_set = IHDP(i, "test")
        elif dataset == "jobs":
            train_set = Jobs(i, "train")
            test_set = Jobs(i, "test")

        # Create model
        if model_name == "cfrnet":
            net = CfrNet(train_set.x.shape[1], config.hidden_dim_rep, config.hidden_dim_rep, config)
        elif model_name == "tarnet":
            net = CfrNet(train_set.x.shape[1], config.hidden_dim_rep, config.hidden_dim_rep, config, use_ipm=False)
        elif model_name == "logistic":
            net = LogisticRegressionNet(train_set.x.shape[1], config)

        start = time()
        train_losses, test_losses = train_test_loop(net, train_set, test_set)
        print(f"Time for realization {i}: {time() - start:.2f}s\nFinal test loss: {test_losses[-1]}")

        results = evaluate(net, train_set, test_set, dataset)
        all_results.append(results)

        # Log results while running
        if config.do_save:
            df = results_to_df(all_results)
            df.to_csv(f"{path}.csv", index=False)
    # plot(train_losses, test_losses)

    df = results_to_df(all_results)
    fig = plot_results(df, dataset)

    if config.do_save:
        plt.savefig(f"{path}.png")
        df.to_csv(f"{path}.csv", index=False)
