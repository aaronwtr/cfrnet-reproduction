
def loss_PEHE(prediction, truth):
    """
    The expected Precision in Estimation of Hetrogeneous Effect
    PEHE function (1) of the paper.
    :param prediction:
    :param truth:
    :return:
    """

    return (prediction - truth) ** 2