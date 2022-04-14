from config import Config
from model import CfrNet
from utils import data_to_device


def train(train_loader, net: CfrNet, config: Config):
    """
    Train the network
    :param train_loader: loading the batches
    :param net: cfr net
    :param config: config class (default Config())
    :return:
    """
    avg_loss = 0

    # iterate through batches
    for i, data in enumerate(train_loader):
        if config.use_gpu:
            data = data_to_device(data, config.device)

        loss = net.calculate_loss(data, config)

        net.optim.zero_grad()
        loss.backward()
        net.optim.step()

        avg_loss += loss

    avg_loss = avg_loss / len(train_loader)
    return avg_loss
