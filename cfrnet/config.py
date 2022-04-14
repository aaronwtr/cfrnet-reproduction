from utils import get_computing_device
import torch.cuda


class Config:
    def __init__(self):
        self.num_epochs = 100

        # Learning rate 1e-4 seems to be better with CfrNet
        # 1e-3 seems to be better with LogisticRegressionNet
        self.learning_rate = 1e-3

        # The rate of the weight decay was not mentioned in the paper.
        self.weight_decay = 1e-5  # Regularization term in pytorch
        
        self.batch_size = 100
        self.split_h = True
        self.ipm_function = 'mmd'  # Use mmd or wasserstein
        self.alpha = 1
        self.n_iterations = 1000
        self.dataset = "ihdp"  # Either jobs or ihdp
        self.do_save = True  # Whether to save the output
        self.do_log_epochs = False  # Whether to log the number of epochs

        # Size of hidden dimensions
        self.hidden_dim_rep = 200
        self.hidden_dim_hypo = 100
        self.prefer_gpu = False  # Set to True if you want to use the GPU
        self.use_gpu = self.prefer_gpu and torch.cuda.is_available()
        self.device = get_computing_device(self.use_gpu)

        # Model to use
        self.model_name = "cfrnet"  # Either logistic, cfrnet, or tarnet

        self.output_dir = "output"
