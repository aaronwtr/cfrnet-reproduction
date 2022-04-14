import torch

def wasserstein_distance(X_t, X_c, t=None, p=0.5, lam=1, iterations=10):
    """
    Implementation of algorithm 2 (appendix B.1)
    :param X_t: mini-batch of treated samples in the form [phi(X),t,y]
    :param X_c: mini-batch of control samples in the form [phi(X),t,y]
    :param t: indication if x is treated or not
    :param p: probability of treated: p = p(t = 1) = sum(t_i) over all i (?)
    :param lam: smoothing parameter (standard 1)
    :param iterations: ? (standard 10)
    :return: Wasserstein distance between treated and control sample batches
    """
    # # (1-2) batch divided into treated and control
    # X_t = X[torch.where(t > 0)]
    # X_c = X[torch.where(t < 1)]

    # batch sizes (should be equal?)
    n_t = X_t.size(dim=0)
    n_c = X_c.size(dim=0)

    if n_c == 0 or n_t == 0:
        return 0

    # (3) compute distance matrix M
    M = torch.tensor([[torch.linalg.vector_norm(X_t[i] - X_c[j])**2 for j in range(n_c)] for i in range(n_t)])

    # dropout (?)
    # drop = torch.nn.Dropout(M, 10./(nt * nc))

    # (4) calculate transport matrix T
    a = p * torch.ones((n_t, 1)) / n_t
    b = (1 - p) * torch.ones((n_c, 1)) / n_c

    K = torch.exp(-lam * M)
    K_tilde = K / a

    u = a
    for i in range(0, iterations):
        u = 1.0 / torch.matmul(K_tilde, b / torch.matmul(torch.transpose(K, 0, 1), u))

    v = b / torch.matmul(torch.transpose(K, 0, 1), u)

    T = u * (torch.transpose(v, 0, 1) * K)

    # (5) calculate distance
    E = T * M

    return 2 * torch.sum(E)
