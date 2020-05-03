import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.categorical import Categorical
import torch_ac_winnie
import utils


class PNNModel(nn.Module, torch_ac_winnie.RecurrentACModel):
    def __init__(self, obs_space, action_space, use_memory=False, use_text=False, use_pnn=False, base=None,
                 base_kwargs=None):
        super(PNNModel, self).__init__()

        if base_kwargs is None:
            base_kwargs = {}
        if base is None:
            if use_pnn:
                base = PNNConvBase
        self.base = base(obs_space, action_space, **base_kwargs)

        # Decide which components are enabled
        self.use_text = use_text
        self.use_memory = use_memory

        n = obs_space["image"][0]
        m = obs_space["image"][1]
        self.image_embedding_size = ((n - 1) // 2 - 2) * ((m - 1) // 2 - 2) * 64

    @property
    def memory_size(self):
        return 2 * self.semi_memory_size

    @property
    def semi_memory_size(self):
        return self.image_embedding_size


class PNNConvBase(nn.Module, torch_ac_winnie.RecurrentACModel):
    def __init__(self, obs_space, action_space):
        super().__init__()
        self.columns = nn.ModuleList([])
        self.obs_space = obs_space
        self.action_space = action_space
        self.n_layers = 4

    # adds a new column to pnn
    def new_task(self):
        new_column = PNNColumn(self.obs_space, self.action_space, use_memory=False, use_text=False)
        self.columns.append(new_column)

    def freeze_columns(self, skip=None):  # freezes the weights of previous columns
        if skip is None:
            skip = []

        for i, c in enumerate(self.columns):
            if i not in skip:
                for params in c.parameters():
                    params.requires_grad = False


class PNNColumn(nn.Module, torch_ac_winnie.RecurrentACModel):
    def __init__(self, obs_space, action_space, use_memory=False, use_text=False):
        super().__init__()

        # Decide which components are enabled
        self.use_text = use_text
        self.use_memory = use_memory
        self.output_shapes = [16, 32, 64, 64]
        self.input_shapes = [3, 16, 32, 64]

        # for initiatiate parameters
        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                               constant_(x, 0), nn.init.calculate_gain('relu'))
        # Define layers
        self.conv1 = init_(nn.Conv2d(3, 16, (2, 2)))
        self.conv2 = init_(nn.Conv2d(16, 32, (2, 2)))
        self.conv3 = init_(nn.Conv2d(32, 64, (2, 2)))
        self.fc = init_(nn.Linear(64, 64))

        # Define operations on layers
        self.mp = nn.MaxPool2d((2, 2))
        self.relu = nn.ReLU()
        self.flatten = Flatten()

        # for correctly calculate size
        n = obs_space["image"][0]
        m = obs_space["image"][1]
        self.image_embedding_size = ((n - 1) // 2 - 2) * ((m - 1) // 2 - 2) * 64

        # Define memory
        if self.use_memory:
            self.memory_rnn = nn.LSTMCell(self.image_embedding_size, self.semi_memory_size)
        # Define text embedding
        if self.use_text:
            self.word_embedding_size = 32
            self.word_embedding = nn.Embedding(obs_space["text"], self.word_embedding_size)
            self.text_embedding_size = 128
            self.text_rnn = nn.GRU(self.word_embedding_size, self.text_embedding_size, batch_first=True)
        # Resize image embedding
        self.embedding_size = self.semi_memory_size
        if self.use_text:
            self.embedding_size += self.text_embedding_size

        # Define actor's model
        self.actor = nn.Sequential(
            nn.Linear(self.embedding_size, 64),
            nn.Tanh(),
            nn.Linear(64, action_space.n)
        )

        # Define critic's model
        self.critic = nn.Sequential(
            nn.Linear(self.embedding_size, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )

        # Initialize parameters correctly
        self.apply(init_params)

    def layers(self, i, x):
        if i == 0:
            return self.mp(self.relu(self.conv1(x)))
        elif i == 1:
            return self.relu(self.conv2(x))
        elif i == 2:
            return self.relu(self.conv3(x))
        elif i == 3:
            return self.relu(self.fc(self.flatten(x)))

    @property
    def memory_size(self):
        return 2 * self.semi_memory_size

    @property
    def semi_memory_size(self):
        return self.image_embedding_size

    def forward(self, x):
        outs = []
        for i in range(4):
            x = self.layers(i, x)
            outs.append(x)
        return outs



def init_params(m):
    classname = m.__class__.__name__
    if classname.find("Linear") != -1:
        m.weight.data.normal_(0, 1)
        m.weight.data *= 1 / torch.sqrt(m.weight.data.pow(2).sum(1, keepdim=True))
        if m.bias is not None:
            m.bias.data.fill_(0)

def init(module, weight_init, bias_init, gain=1):
    weight_init(module.weight.data, gain=gain)
    bias_init(module.bias.data)
    return module

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)
