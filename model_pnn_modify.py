import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.categorical import Categorical
import torch_ac_pnn
import numpy as np
import utils


class PNNModel(nn.Module, torch_ac_pnn.RecurrentACModel):
    def __init__(self, obs_space, action_space, use_memory=False, use_text=False, use_pnn=False, base=None):
        super(PNNModel, self).__init__()
        self.obs_space = obs_space
        self.action_space = action_space
        self.use_memory = use_memory
        self.use_text = use_text

        # for correctly calculate size
        n = self.obs_space["image"][0]
        m = self.obs_space["image"][1]
        self.image_embedding_size = ((n - 1) // 2 - 2) * ((m - 1) // 2 - 2) * 64
        # self.train()  # what does this do?

        if base is None:
            if use_pnn:
                base = PNNConvBase
        self.base = base(self.obs_space, self.action_space, self.use_memory, self.use_text)

    @property
    def memory_size(self):
        return 2 * self.semi_memory_size

    @property
    def semi_memory_size(self):
        return self.image_embedding_size


class PNNConvBase(nn.Module, torch_ac_pnn.RecurrentACModel):
    def __init__(self, obs_space, action_space, use_memory=False, use_text=False):
        super().__init__()
        self.columns = nn.ModuleList([])
        self.n_layers = 5

        self.alpha = nn.ModuleList([])
        self.V = nn.ModuleList([])
        self.U = nn.ModuleList([])

        self.obs_space = obs_space
        self.action_space = action_space
        self.use_memory = use_memory
        self.use_text = use_text

        # for correctly calculate size
        n = self.obs_space["image"][0]
        m = self.obs_space["image"][1]
        self.image_embedding_size = ((n - 1) // 2 - 2) * ((m - 1) // 2 - 2) * 64
        self.train()  # what does this do?

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

        self.output_shapes = [16, 32, 64, self.embedding_size, 64]
        self.input_shapes = [3, 16, 32, 64, self.embedding_size]
        self.V_out_shapes = [int(3 / 2), int(16 / 2), int(32 / 2), int(64 / 2), int(self.embedding_size / 2),
                             [int(action_space.n / 2), 1]]
        #self.V_out_shapes = [3, 16, 32, 64, self.embedding_size, [action_space.n, 1]]

    @property
    def memory_size(self):
        return 2 * self.semi_memory_size

    @property
    def semi_memory_size(self):
        return self.image_embedding_size

    def _get_embed_text(self, text):
        _, hidden = self.text_rnn(self.word_embedding(text))
        return hidden[-1]

    def memory_text(self, obs, x):

        # if self.use_memory:
        #    hidden = (memory[:, :self.semi_memory_size], memory[:, self.semi_memory_size:])
        #   hidden = self.memory_rnn(obs, hidden)
        #  embedding = hidden[0]
        #   memory = torch.cat(hidden, dim=1)
        # else:
        embedding = x

        if self.use_text:
            embed_text = self._get_embed_text(obs.text)
            embedding = torch.cat((embedding, embed_text), dim=1)  # concat image input and text input

        return embedding

    def generate_v(self, pre_layer, map_in_shape, map_out_shape):
        # here, map in shape is the shape of a_h or previous column, previous layer
        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                               constant_(x, 0))

        # specify the dimension reduction conv layer v
        if pre_layer == 0:
            v = init_(nn.Conv2d(map_in_shape, map_out_shape,
                                1))  # take output from  layer 0, previous column, do dimensionality reduction
        elif pre_layer == 1:
            v = init_(nn.Conv2d(map_in_shape, map_out_shape,
                                1))  # take output from  layer 1, previous column, do dimensionality reduction
        elif pre_layer == 2:
            v = init_(nn.Conv2d(map_in_shape, map_out_shape,
                                1))  # take output from  layer 2, previous column, do dimensionality reduction
        else:
            v = init_(nn.Linear(map_in_shape, map_out_shape))
        return v

    def generate_U(self, cur_layer, map_in_shape, map_out_shape):

        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                               constant_(x, 0))

        if cur_layer == 1:  # process in the same way as conv2
            u = init_(nn.Conv2d(map_in_shape, map_out_shape, (2, 2)))
        elif cur_layer == 2:  # process in the same way as conv3
            u = init_(nn.Conv2d(map_in_shape, map_out_shape, (2, 2)))
        elif cur_layer == 3:
            u = init_(nn.Linear(map_in_shape, map_out_shape))
        else:
            u = init_(nn.Linear(map_in_shape, map_out_shape))
        return u

    def forward(self, obs, memory):  # what is rnn_hxs? what is masks????????
        assert self.columns, 'PNN should at least have one column (missing call to `new_task` ?)'
        # x = (x / 255.0)  # why x is devided by 255 (from srikar)????????????
        # output_column = []
        x = obs.image.transpose(1, 3).transpose(2, 3)
        inputs = [self.columns[i].layers(0, obs, x, memory) for i in range(len(self.columns))]
        for l in range(1, self.n_layers):
            output_column = [self.columns[0].layers(l, obs, inputs[0], memory)]


            for c in range(1, len(self.columns)):
                # in_cur = self.columns[c].layers(0, obs, x, memory)
                # in_pre = self.columns[c-1].layers(0, obs, x, memory)

                # out_cur = self.columns[c].layers(l, obs, in_cur, memory)   # output from current column, current layer
                #if l == 4:
                    # pre_col = inputs[c - 1][0]  # output form previous column, previous layer
                    # memory_pre = inputs[c - 1][1]   # check how to solve this
                    # memory = memory + memory_pre*0
                    #pre_col = inputs[c - 1]
                    #out_cur = self.columns[c].layers(l, obs, inputs[c],
                                                     #memory)  # output from current column, current layer
                    #out_cur_act = out_cur[0]
                    #out_cur_cri = out_cur[1]
                #elif l == 5:
                    #pre_col = inputs[c - 1]

                pre_col = inputs[c - 1]
                out_cur = self.columns[c].layers(l, obs, inputs[c], memory)  # output from current column, current layer

                # Scaling, dimensionality reduction and forwarding
                #  a for controlling the strength of scaling layer

                if l == 3:  # text layer
                    a = self.alpha[c - 1][l - 1]
                    a_h = a(pre_col)  # pre_col = 2 ==> output from conv3
                    V = self.V[c - 1][l-1]
                    V_a_h = F.relu(V(a_h))
                    V_a_h = V_a_h.reshape(V_a_h.shape[0], -1)  # reshape
                    # V_a_h = self.columns[c-1].layers(l, obs, V_a_h, memory)  #thick about it
                    # V_a_h = self.memory_text(obs, V_a_h, memory)
                    # memory = U_V_a_h[0]  # memory memory should just be memory form current column, check later
                    U = self.U[c - 1][l-1]
                    U_V_a_h = U(V_a_h)

                #elif l == 4:  # actor_critic layer
                 #   a = self.alpha[c - 1][l - 1]
                  #  a_h = a(pre_col)
                   # V_a_h = F.relu(V(a_h))

                    #U_act = self.U[c - 1][l-1]
                    #U_cri = self.U[c - 1][l]
                    #U_V_a_h_act = U_act(V_a_h)
                    #U_V_a_h_cri = U_cri(V_a_h)

                #elif l == 5:  # final layer
                 #   pass
                    #a_act = self.alpha[c - 1][l - 1]
                    #a_h_act = a_act(pre_col[0])
                    #a_cri = self.alpha[c - 1][l]
                    #a_h_cri = a_cri(pre_col[1])
                    #V_act = self.V[c - 1][l-1]
                    #V_cri = self.V[c - 1][l]
                    #V_a_h_act = F.relu(V_act(a_h_act))
                    #V_a_h_cri = F.relu(V_cri(a_h_cri))

                    #U_act = self.U[c - 1][l]
                    #U_cri = self.U[c - 1][l+1]
                    #U_V_a_h_act = U_act(V_a_h_act)
                    #U_V_a_h_cri = U_cri(V_a_h_cri)
                    #U_V_a_h = [U_V_a_h_act, U_V_a_h_cri]

                else:
                    a = self.alpha[c - 1][l - 1]
                    a_h = a(pre_col)
                    V = self.V[c - 1][l-1]
                    V_a_h = F.relu(V(a_h))
                    U = self.U[c - 1][l-1]
                    U_V_a_h = U(V_a_h)

                # combine output from previous and current column
                #if l == 4:
                    #out_act = F.relu(out_cur_act + U_V_a_h_act)
                    #out_cri = F.relu(out_cur_cri + U_V_a_h_cri)
                    #out = [out_act, out_cri]
                #elif l == 5:
                    #inputs_final = inputs[c]
                    #inputs_final = U_V_a_h + inputs[c]
                    #out = self.columns[c].layers(l, obs, inputs_final, memory)
                #else:
                out = F.relu(out_cur + U_V_a_h)

                output_column.append(out)

            inputs = output_column

        # for predicting value and action
        output = inputs[-1]  # take last column output

        dist, value, memory = self.columns[-1].layers(5, obs, output, memory)
        return dist, value, memory

    # adds a new column to pnn
    def new_task(self):
        new_column = PNNColumn(self.obs_space, self.action_space, self.use_memory, self.use_text)
        self.columns.append(new_column)

        if len(self.columns) > 1:

            a_list = []
            V_list = []
            U_list = []
            for l in range(1, self.n_layers):
                #if l == 5:
                    #a_act = ScaleLayer(0.01)
                    #a_cri = ScaleLayer(0.01)
                    #V_act = self.generate_v(l - 1,  self.input_shapes[l][0], self.V_out_shapes[l][0])
                    #V_cri = self.generate_v(l - 1,  self.input_shapes[l][1], self.V_out_shapes[l][1])
                    #pass

                a = ScaleLayer(0.01)
                V = self.generate_v(l - 1, self.input_shapes[l], self.V_out_shapes[l])

                #if l == 4:
                 #   U_act = self.generate_U(l, self.V_out_shapes[l], self.output_shapes[l][0])
                  #  U_cri = self.generate_U(l, self.V_out_shapes[l], self.output_shapes[l][1])
                #elif l == 5:
                    #U_act = self.generate_U(l, self.V_out_shapes[l][0], self.output_shapes[l][0])
                    #U_cri = self.g/enerate_U(l, self.V_out_shapes[l][1], self.output_shapes[l][1])
                    #pass

                U = self.generate_U(l, self.V_out_shapes[l], self.output_shapes[l])

                #if l == 4:
                 #   a_list.append(a)
                  #  V_list.append(V)
                   # U_list.append(U_act)
                    #U_list.append(U_cri)

                #elif l == 5:
                    #a_list.append(a_act)
                    #a_list.append(a_cri)
                    #V_list.append(V_act)
                    #V_list.append(V_cri)
                    #U_list.append(U_act)
                    #U_list.append(U_cri)
                    #pass
                #else:
                a_list.append(a)
                V_list.append(V)
                U_list.append(U)

            a_list = nn.ModuleList(a_list)
            V_list = nn.ModuleList(V_list)
            U_list = nn.ModuleList(U_list)

            self.alpha.append(a_list)
            self.V.append(V_list)
            self.U.append(U_list)

    def freeze_columns(self, skip=None):  # freezes the weights of previous columns
        if skip is None:
            skip = []

        for i, c in enumerate(self.columns):
            if i not in skip:
                for params in c.parameters():
                    params.requires_grad = False

    def parameters(self, col=None):
        if col is None:
            return super(PNNConvBase, self).parameters()

        return self.columns[col].parameters()


class PNNColumn(nn.Module, torch_ac_pnn.RecurrentACModel):
    def __init__(self, obs_space, action_space, use_memory, use_text):
        super().__init__()

        # for initiatiate parameters
        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                               constant_(x, 0), nn.init.calculate_gain('relu'))
        # Define layers
        self.conv1 = init_(nn.Conv2d(3, 16, (2, 2)))
        self.conv2 = init_(nn.Conv2d(16, 32, (2, 2)))
        self.conv3 = init_(nn.Conv2d(32, 64, (2, 2)))

        # Define operations on layers
        self.mp = nn.MaxPool2d((2, 2))
        self.relu = nn.ReLU()

        ############################# From original ppo code ################################
        # Decide which components are enabled
        self.use_text = use_text
        self.use_memory = use_memory

        # for correctly calculate size
        n = obs_space["image"][0]
        m = obs_space["image"][1]
        self.image_embedding_size = ((n - 1) // 2 - 2) * ((m - 1) // 2 - 2) * 64

        # Define memory
        # if self.use_memory:
        #   self.memory_rnn = nn.LSTMCell(self.image_embedding_size, self.semi_memory_size)

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

        # define fully connected layer
        self.fc = nn.Sequential(nn.Linear(self.embedding_size, 64))

        # Define actor's model
        self.actor = nn.Sequential(
            #nn.Linear(64, 64),
            #nn.Tanh(),
            nn.Linear(64, action_space.n)
        )

        # Define critic's model
        self.critic = nn.Sequential(
            #nn.Linear(64, 64),
            #nn.Tanh(),
            nn.Linear(64, 1)
        )

        # Initialize parameters correctly
        self.apply(init_params)

    @property
    def memory_size(self):
        return 2 * self.semi_memory_size

    @property
    def semi_memory_size(self):
        return self.image_embedding_size

    def memory_text(self, obs, x):

        # if self.use_memory:
        # hidden = (memory[:, :self.semi_memory_size], memory[:, self.semi_memory_size:])
        # hidden = self.memory_rnn(obs, hidden)
        # embedding = hidden[0]
        # memory = torch.cat(hidden, dim=1)

        embedding = x

        if self.use_text:
            embed_text = self._get_embed_text(obs.text)
            embedding = torch.cat((embedding, embed_text), dim=1)  # concat image input and text input

        return embedding

    # define pnn column layer structure
    def layers(self, i, obs, x, memory):
        if i == 0:  # conv1
            return self.mp(self.relu(self.conv1(x)))
        elif i == 1:  # conv2
            return self.relu(self.conv2(x))
        elif i == 2:  # conv3
            return self.relu(self.conv3(x))
        elif i == 3:  # memory text
            # connect to text/memory layer
            x = x.reshape(x.shape[0], -1)
            return self.memory_text(obs, x)
        elif i == 4:  # fc layer
            x = self.fc(x)
            return x
        else: # actor and crtic layer
            x_a = self.actor(x)
            dist = Categorical(logits=F.log_softmax(x_a))

            x_c = self.critic(x)
            value = x_c.squeeze(1)
            return [dist, value, memory]

    # forward method for layers in one PNN column
    def forward(self, obs, memory):
        x = obs.image.transpose(1, 3).transpose(2, 3)
        for i in range(5):  # three convolution layers
            x = self.layers(i, obs, x, memory)

        dist, value, memory = self.layers(5, obs, x, memory)
        return [dist, value, memory]  # check later,outs is for Srikar, dist, value and memory is for original ppo

    def _get_embed_text(self, text):
        _, hidden = self.text_rnn(self.word_embedding(text))
        return hidden[-1]


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


class ScaleLayer(nn.Module):

    def __init__(self, init_value=0.01):
        super().__init__()
        self.scale = nn.Parameter(torch.FloatTensor([init_value]))

    def forward(self, x):
        return x * self.scale


#########################################################################
class ACModel(nn.Module, torch_ac_pnn.RecurrentACModel):
    def __init__(self, obs_space, action_space, use_memory=False, use_text=False):
        super().__init__()

        # Decide which components are enabled
        self.use_text = use_text
        self.use_memory = use_memory

        # Define image embedding
        self.image_conv = nn.Sequential(
            nn.Conv2d(3, 16, (2, 2)),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),
            nn.Conv2d(16, 32, (2, 2)),
            nn.ReLU(),
            nn.Conv2d(32, 64, (2, 2)),
            nn.ReLU()
        )
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
            nn.Linear(self.embedding_size, 64),  #
            nn.Tanh(),
            nn.Linear(64, action_space.n)
        )

        # Define critic's model
        self.critic = nn.Sequential(
            nn.Linear(self.embedding_size, 64),  # self.embedding_size
            nn.Tanh(),
            nn.Linear(64, 1)
        )

        # Initialize parameters correctly
        self.apply(init_params)

    @property
    def memory_size(self):
        return 2 * self.semi_memory_size

    @property
    def semi_memory_size(self):
        return self.image_embedding_size

    def forward(self, obs, memory):
        x = obs.image.transpose(1, 3).transpose(2, 3)
        x = self.image_conv(x)
        x = x.reshape(x.shape[0], -1)

        if self.use_memory:
            hidden = (memory[:, :self.semi_memory_size], memory[:, self.semi_memory_size:])
            hidden = self.memory_rnn(x, hidden)
            embedding = hidden[0]
            memory = torch.cat(hidden, dim=1)
        else:
            embedding = x

        if self.use_text:
            embed_text = self._get_embed_text(obs.text)
            embedding = torch.cat((embedding, embed_text), dim=1)

        x = self.actor(embedding)
        dist = Categorical(logits=F.log_softmax(x, dim=1))

        x = self.critic(embedding)
        value = x.squeeze(1)

        return dist, value, memory

    def _get_embed_text(self, text):
        _, hidden = self.text_rnn(self.word_embedding(text))
        return hidden[-1]
