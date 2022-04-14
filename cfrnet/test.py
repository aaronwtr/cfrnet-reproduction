import torch

from config import Config
from model import CfrNet
from utils import data_to_device


def test(test_loader, net: CfrNet, config: Config):
    """
    Test the network
    :param train_loader: loading the batches
    :param net: cfr net
    :param config: config class (default Config())
    :return:
    """
    avg_loss = 0

    # iterate through batches
    with torch.no_grad():
        for i, data in enumerate(test_loader):
            if config.use_gpu:
                data = data_to_device(data, config.device)

            loss = net.calculate_loss(data, config)
            avg_loss += loss

    avg_loss = avg_loss / len(test_loader)
    return avg_loss
