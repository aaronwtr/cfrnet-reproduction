from matplotlib import pyplot as plt
import numpy as np

def plot(train_losses, test_losses):
    plt.plot(np.arange(len(train_losses)), train_losses, label="Train")
    plt.plot(np.arange(len(test_losses)), test_losses, label="Test")
    plt.legend()
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.show()

def plot_results(results, dataset):
    fig, (ax1, ax2) = plt.subplots(1, 2)

    if dataset == "ihdp":
        ax1.plot(np.arange(len(results['ate_error'])), results['ate_error'])
        ax1.set_title("ATE error")
        ax2.plot(np.arange(len(results['pehe_error'])), results['pehe_error'])
        ax2.set_title("PEHE error")
    elif dataset == "jobs":
        ax1.plot(np.arange(len(results['att_error'])), results['att_error'])
        ax1.set_title("ATT error")
        ax2.plot(np.arange(len(results['r_pol'])), results['r_pol'])
        ax2.set_title("Policy risk")

    # plt.show()
    return fig