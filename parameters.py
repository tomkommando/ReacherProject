import torch


class Params:
    def __init__(self):

        # output folders
        self.WEIGHTS_FOLDER = "./outputs/"

        # GPU
        self.DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # Training Process

        self.N_EPISODES = 10000  # max episodes
        self.MAX_T = 1200  # max steps per episode

        # Agent
        self.AGENT_SEED = 0  # random seed for agent
        self.BUFFER_SIZE = int(1e4)  # replay buffer size
        self.BATCH_SIZE = 128  # minibatch size
        self.GAMMA = 0.99  # discount factor
        self.TAU = 1e-3  # interpolation parameter for soft update of target parameters
        self.LR_ACTOR = 1e-3  # learning rate of the actor
        self.LR_CRITIC = 1e-3  # learning rate of the critic
        self.WEIGHT_DECAY = 0  # L2 weight decay

        # Ornstein-Uhlenbeck Process
        self.MU = 0.  # average
        self.THETA = 0.15  # drift
        self.SIGMA = 0.2  # volatility

        # Network
        self.NN_SEED = 0  # random seed for networks
        self.FC1_UNITS_ACTOR = 64  # size of first hidden layer, actor
        self.FC2_UNITS_ACTOR = 128  # size of second hidden layer, actor

        self.FC1_UNITS_CRITIC = 64  # size of first hidden layer, critic
        self.FC2_UNITS_CRITIC = 128  # size of second hidden layer, critic
