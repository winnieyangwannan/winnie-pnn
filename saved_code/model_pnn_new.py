import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.categorical import Categorical
import torch_ac_winnie
import numpy as np
import utils


class PNNModel(nn.Module, torch_ac_winnie.RecurrentACModel):
    def __init__(self, obs_space, action_space, use_memory=False, use_text=False, use_pnn=False, base=None):
        super(PNNModel, self).__init__()

        if base is None:
            if use_pnn:
                base = PNNConvBase
        self.base = base(obs_space, action_space, use_memory=False, use_text=False)


class PNNConvBase(nn.Module, torch_ac_winnie.RecurrentACModel):
    def __init__(self, obs_space, action_space, use_memory=False, use_text=False):
        super().__init__()
        self.columns = nn.ModuleList([])
        self.n_layers = 6  # winnie choutput_columnange to incorporate text and memory
        #self.output_column.append = []

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
        #self.train()  # what does this do?

    @property
    def memory_size(self):
        return 2 * self.semi_memory_size

    @property
    def semi_memory_size(self):
        return self.image_embedding_size

    def _get_embed_text(self, text):
        _, hidden = self.text_rnn(self.word_embedding(text))
        return hidden[-1]

    def memory_text(self, obs, x, memory):

        if self.use_memory:
            hidden = (memory[:, :self.semi_memory_size], memory[:, self.semi_memory_size:])
            hidden = self.memory_rnn(obs, hidden)
            embedding = hidden[0]
            memory = torch.cat(hidden, dim=1)
        else:
            embedding = x

        if self.use_text:
            embed_text = self._get_embed_text(obs.text)
            embedding = torch.cat((embedding, embed_text), dim=1) # concat image input and text input

        return [embedding, memory]

    def generate_v(self, cur_col, pre_layer, map_in):
        # here, map in shape is the shape of a_h or previous column, previous layer
        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                               constant_(x, 0))

        map_in_shape = map_in.shape[1]
        map_out_shape = int(map_in_shape / (2 * cur_col))

        if map_out_shape == 0:
            map_out_shape = 1
        else:
            pass

        # specify the dimension reduction conv layer v
        if pre_layer == 0:
            v = init_(nn.Conv2d(map_in_shape, map_out_shape, 1))  # take output from  layer 0, previous column, do dimensionality reduction
        elif pre_layer == 1:
            v = init_(nn.Conv2d(map_in_shape, map_out_shape, 1))  # take output from  layer 1, previous column, do dimensionality reduction
        elif pre_layer == 2:
            v = init_(nn.Conv2d(map_in_shape, map_out_shape, 1))  # take output from  layer 2, previous column, do dimensionality reduction
        elif pre_layer == 3:
            v = init_(nn.Linear(map_in_shape, map_out_shape))
        elif pre_layer == 4:
            v = init_(nn.Linear(map_in_shape, map_out_shape))

        return v

    def generate_U(self, cur_layer, map_in_shape, map_out_shape):
        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                               constant_(x, 0))

        if cur_layer == 1:  #  process in the same way as conv2
            u = init_(nn.Conv2d(map_in_shape, map_out_shape, (2, 2)))
        elif cur_layer == 2:  # process in the same way as conv3
            u = init_(nn.Conv2d(map_in_shape, map_out_shape, (2, 2)))
        elif cur_layer == 3:
            u = init_(nn.Linear(map_in_shape, map_out_shape))
        elif cur_layer == 4:
            u = init_(nn.Linear(map_in_shape, map_out_shape))
        elif cur_layer == 5:
            u = init_(nn.Linear(map_in_shape, map_out_shape))
        return u

    def forward(self, obs, memory):  # what is rnn_hxs? what is masks????????
        assert self.columns, 'PNN should at least have one column (missing call to `new_task` ?)'
        # x = (x / 255.0)  # why x is devided by 255 (from srikar)????????????
        #output_column = []

        #if len(self.columns) > 1:
        a_list = []
        V_list = []
        U_list = []
        out_column = []
        x = obs.image.transpose(1, 3).transpose(2, 3)
        input_pre = self.columns[0].layers(0, obs, x, memory)

        for l in range(1, self.n_layers):
            # output from layer column 0
            if l == 3:
                out = self.columns[0].layers(l, obs, input_pre, memory)
                memory = out[1]
                out = out[0]
            else:
                out = self.columns[0].layers(l, obs, input_pre, memory)
            #if len(self.columns) == 1:

            inputs = [self.columns[i].layers(0, x) for i in range(len(self.columns))]
            for c in range(1, len(self.columns)):
                #in_cur = self.columns[c].layers(0, obs, x, memory)
                #in_pre = self.columns[c-1].layers(0, obs, x, memory)

                #out_cur = self.columns[c].layers(l, obs, in_cur, memory)   # output from current column, current layer

                # output from current column
                input_cur = self.columns[1].layers(0, obs, x, memory)
                if l == 4:
                    pre_col = input_pre  # output form previous column, previous layer
                    #memory_pre = inputs[c - 1][1]   # check how to solve this
                    #memory = memory + memory_pre*0
                    out_cur = self.columns[c].layers(l, obs, input_cur, memory)  # output from current column, current layer
                elif l == 5:
                    pre_col = input_pre
                    out_cur = []
                else:
                    pre_col = input_pre
                    out_cur = self.columns[c].layers(l, obs, input_cur, memory)  # output from current column, current layer

                # for actor_critic and memory_text, deal with two outputs separately
                if l == 3:
                    memory = out_cur[1]
                    out_cur = out_cur[0]

                elif l == 4:
                    out_cur_act = out_cur[0]
                    out_cur_cri = out_cur[1]
                else:
                    pass


                # Scaling, dimensionality reduction and forwarding
                #  a for controlling the strength of scaling layer
                a = ScaleLayer(0.01)
                if l == 3: # text layer
                    a_h = a(pre_col)  # pre_col = 2 ==> output from conv3
                    V = self.generate_v(c, l-1, a_h)
                    V_a_h = F.relu(V(a_h))
                    V_a_h = V_a_h.reshape(V_a_h.shape[0], -1)  # reshape
                    #V_a_h = self.columns[c-1].layers(l, obs, V_a_h, memory)  #thick about it
                    #V_a_h = self.memory_text(obs, V_a_h, memory)
                    # memory = U_V_a_h[0]  # memory memory should just be memory form current column, check later
                    U = self.generate_U(l, V_a_h.shape[1], out_cur.shape[1])
                    U_V_a_h = U(V_a_h)

                elif l == 4:  # actor_critic layer
                    a_h = a(pre_col)  # ONLY output from text_memory layer NOT memory transfer to current column
                    V = self.generate_v(c, l - 1, a_h)
                    V_a_h = F.relu(V(a_h))

                    U_act = self.generate_U(l, V_a_h.shape[1], out_cur_act.shape[1])
                    U_cri = self.generate_U(l, V_a_h.shape[1], out_cur_cri.shape[1])
                    U_V_a_h_act = U_act(V_a_h)
                    U_V_a_h_cri = U_cri(V_a_h)
                    #U = [U_act, U_cri]

                elif l == 5:  # final layer
                    a_h_act = a(pre_col[0])
                    a_h_cri = a(pre_col[1])
                    V_act = self.generate_v(c, l - 1, a_h_act)
                    V_cri = self.generate_v(c, l - 1, a_h_cri)
                    #V = [V_act, V_cri]
                    V_a_h_act = F.relu(V_act(a_h_act))
                    V_a_h_cri = F.relu(V_cri(a_h_cri))

                    U_act = self.generate_U(l, V_a_h_act.shape[1], out_cur_act.shape[1])
                    U_cri = self.generate_U(l, V_a_h_cri.shape[1], out_cur_cri.shape[1])
                    U_V_a_h_act = U_act(V_a_h_act)
                    U_V_a_h_cri = U_cri(V_a_h_cri)
                    U_V_a_h = [U_V_a_h_act, U_V_a_h_cri]
                    #U_V_a_h_cri = U_V_a_h_cri.squeeze(1)
                    #U = [U_act, U_cri]

                else:
                    a_h = a(pre_col)
                    V = self.generate_v(c, l-1, a_h)
                    V_a_h = F.relu(V(a_h))
                    U = self.generate_U(l, V_a_h.shape[1], out_cur.shape[1])
                    U_V_a_h = U(V_a_h)


                # combine output from previous and current column
                if l == 4:
                    out_act = F.relu(out_cur_act + U_V_a_h_act)
                    out_cri = F.relu(out_cur_cri + U_V_a_h_cri)
                    out = [out_act, out_cri]
                elif l == 5:

                    input_final = U_V_a_h + input_pre
                    out = self.columns[c].layers(l, obs, input_final, memory)
                else:
                    out = F.relu(out_cur + U_V_a_h)

                #output_column.append(out)
                if l == 4:
                    a_list.append(a)
                    V_list.append(V)
                    U_list.append(U_act)
                    U_list.append(U_cri)

                elif l == 5:
                    a_list.append(a)
                    V_list.append(V_act)
                    V_list.append(V_cri)
                    U_list.append(U_act)
                    U_list.append(U_cri)
                else:
                    a_list.append(a)
                    V_list.append(V)
                    U_list.append(U)

                input_pre = out
                out_column.append(out)
            inputs = out_column

        #if len(self.columns) > 1:
        a_list = nn.ModuleList(a_list)
        V_list = nn.ModuleList(V_list)
        U_list = nn.ModuleList(U_list)

        self.alpha.append(a_list)
        self.V.append(V_list)
        self.U.append(U_list)

        # for predicting value and action
        output = out  # take last column output
        dist = output[0]
        value = output[1]
        memory = output[2]
        return dist, value, memory

    # adds a new column to pnn
    def new_task(self):
        new_column = PNNColumn(self.obs_space, self.action_space, self.use_memory, self.use_text)
        self.columns.append(new_column)

        #init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                               #constant_(x, 0))

        #if len(self.columns) > 1:
         #   a = ScaleLayer(0.01)

            #################################################
          #  pre_col, col = self.columns[-2], self.columns[-1]

            #a_list = []
            #V_list = []
            #U_list = []
            #for l in range(1, self.n_layers):


             #   map_in = pre_col.output_shapes[l - 1]
              #  map_out = int(map_in / 2)


               # if l != self.n_layers - 2:  # conv -> conv, tex_memory layer

                #    cur_out = col.output_shapes[l]
                 #   size, stride = pre_col.topology[l - 1]
                  #  u = init_(nn.Conv2d(map_out, cur_out, size, stride=stride))

                #else: # non-conv layers
                 #   input_size = int(col.input_shapes[l] / 2)  # check here, the original code is get v_a_h.shape here
                  #  out_put_size = self.topology(l)

                   # u = init_(nn.Linear(input_size, out_put_size))

                #a_list.append(a)
                #V_list.append(v)
                #U_list.append(u)

            #a_list = nn.ModuleList(a_list)
            #V_list = nn.ModuleList(V_list)
            #U_list = nn.ModuleList(U_list)

            #self.alpha.append(a_list)
            #self.V.append(V_list)
            #self.U.append(U_list)

    def freeze_columns(self, skip=None):  # freezes the weights of previous columns
        if skip is None:
            skip = []

        for i, c in enumerate(self.columns):
            if i not in skip:
                for params in c.parameters():
                    params.requires_grad = False


class PNNColumn(nn.Module, torch_ac_winnie.RecurrentACModel):
    def __init__(self, obs_space, action_space, use_memory, use_text):
        super().__init__()

        # Decide which components are enabled
        self.use_text = use_text
        self.use_memory = use_memory

        # for correctly calculate size
        n = obs_space["image"][0]
        m = obs_space["image"][1]
        self.image_embedding_size = ((n - 1) // 2 - 2) * ((m - 1) // 2 - 2) * 64


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
        self.flatten = Flatten()


        ############################# From original ppo code ################################

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

        # output and input shape ==> check where are they used? Answer: used for add new task, for defining shapes of a few things
        #self.output_shapes = [16, 32, 64, self.embedding_size, action_space.n]
        #self.input_shapes = [3, 16, 32, 64, self.embedding_size]

       # self.topology = [
        #    [2, 1],
         #   [2, 1],
          #  self.embedding_size,
           # [action_space.n,1]
            #]
        # Initialize parameters correctly
        #self.apply(init_params)

    @property
    def memory_size(self):
        return 2 * self.semi_memory_size

    @property
    def semi_memory_size(self):
        return self.image_embedding_size

    def memory_text(self, obs, x, memory):

        if self.use_memory:
            hidden = (memory[:, :self.semi_memory_size], memory[:, self.semi_memory_size:])
            hidden = self.memory_rnn(obs, hidden)
            embedding = hidden[0]
            memory = torch.cat(hidden, dim=1)
        else:
            embedding = x

        if self.use_text:
            embed_text = self._get_embed_text(obs.text)
            embedding = torch.cat((embedding, embed_text), dim=1)  # concat image input and text input

        return [embedding, memory]

    # define pnn column layer structure
    def layers(self, i, obs, x, memory):
        if i == 0: # conv1
            return self.mp(self.relu(self.conv1(x)))
        elif i == 1: # conv2
            return self.relu(self.conv2(x))
        elif i == 2: # conv3
            return self.relu(self.conv3(x))
        elif i == 3:
            # connect to text/memory layer
            x = x.reshape(x.shape[0], -1)
            return self.memory_text(obs, x, memory)
        elif i == 4:
            # actor and crtic layer
            x_a = self.actor(x)
            x_c = self.critic(x)
            return [x_a, x_c]
        elif i == 5:
            dist = Categorical(logits=F.log_softmax(x[0], dim=1))
            value = x[1].squeeze(1)
            return [dist, value, memory]

    # forward method for layers in one PNN column
    def forward(self, obs, memory):
        x = obs.image.transpose(1, 3).transpose(2, 3)
        for i in range(3): # three convolution layers
            x = self.layers(i, obs, x, memory)

        embedding_memory = self.layers(3, obs, x, memory) # memory_text layer
        embedding = embedding_memory[0]
        memory = embedding_memory[1]
        x = embedding

        x = self.layers(4, obs, x, memory) # actor_critic layer
        x = self.layer(5, obs, x, memory) # last layer
        dist = x[0]
        value = x[1]
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
class ACModel(nn.Module, torch_ac_winnie.RecurrentACModel):
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
            nn.Linear(self.embedding_size, 64), #
            nn.Tanh(),
            nn.Linear(64, action_space.n)
        )

        # Define critic's model
        self.critic = nn.Sequential(
            nn.Linear(self.embedding_size, 64), # self.embedding_size
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
