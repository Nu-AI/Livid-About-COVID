import numpy as np
import torch
from torch.nn.parameter import Parameter


###################### Defining Model #######################
#############################################################
class SIRNet(torch.nn.Module):
    def __init__(self, input_size=6, i0=5.6e-6, update_k=True, hidden_size=3,
                 output_size=1, b_lstm=False):
        super(SIRNet, self).__init__()

        assert input_size == 6, 'Input dimension must be 6'  # for now
        assert hidden_size == 3, 'Hidden dimension must be 3'  # for now
        assert output_size == 1, 'Output dimension must be 1'  # for now

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.b_lstm = b_lstm

        # Initializations (from prior training)
        b_init = torch.tensor(
            [[7.3690e-02, 1.0000e-04, 1.0000e-04, 6.5169e-02, 1.4331e-01,
              2.9631e-03]], dtype=torch.float32
        )
        k_init = torch.tensor([[.09]], dtype=torch.float32)

        self.k = Parameter(k_init, requires_grad=update_k)
        self.i0 = i0

        if b_lstm:
            self.i2b = torch.nn.LSTM(input_size, 1, bias=False)
        else:
            self.i2b = torch.nn.Linear(input_size, 1, bias=False)
            self.i2b.weight.data = b_init

        # Output layer sums I and R (total cases)
        self.h2o = torch.nn.Linear(hidden_size, output_size, bias=False)
        self.h2o.weight.data = torch.from_numpy(
            np.array([1, 1, 0]).astype(np.float32).reshape((1, hidden_size)))
        for p in self.h2o.parameters():
            p.requires_grad = False

    def forward(self, X):
        time_steps = X.size(0)  # time first
        batch_size = X.size(1)  # batch second
        hidden = torch.zeros(
            batch_size, self.hidden_size
        ).to(device=X.device)  # hidden state is i,r,s
        hidden[:, 0] = self.i0  # forward should probably take this as an input
        hidden[:, 2] = 1.0 - self.i0
        p = hidden.clone()  # init previous state
        outputs = []
        hiddens = []
        if self.b_lstm:
            # LSTM states
            h_t = torch.zeros(1, 1, 1)
            c_t = torch.zeros(1, 1, 1)
        for t in range(time_steps):
            # contact rate as a function of our input vector
            if self.b_lstm:
                b, (h_t, c_t) = self.i2b(X[None, t], (h_t, c_t))
                b = b.squeeze()
            else:
                b = torch.clamp(self.i2b(X[t]),
                                0)  # contact rate cannot go negative

            # update the hidden state SIR model
            drdt = self.k * p[:, 0]
            hidden[:, 0] = p[:, 0] + p[:, 0] * b * p[:, 2] - drdt  # infected
            hidden[:, 1] = p[:, 1] + drdt  # recovered
            hidden[:, 2] = 1.0 - p[:, 0] - p[:, 1]  # susceptible

            # update the output
            p = hidden.clone()
            output = self.h2o(p)
            outputs.append(output)
            hiddens.append(p)

        return torch.stack(hiddens), torch.stack(outputs)


###################### Defining Model #######################
#############################################################
class SEIRNet(torch.nn.Module):
    def __init__(self, input_size=6, e0=5.6e-6, i0=5.2e-6, update_k=True,
                 hidden_size=4, output_size=1, b_lstm=False,
                 lstm_hidden_size=6):
        super(SEIRNet, self).__init__()

        assert input_size == 6, 'Input dimension must be 6'  # for now
        assert hidden_size == 4, 'Hidden dimension must be 4'  # for now
        assert output_size == 1, 'Output dimension must be 1'  # for now

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.b_lstm = b_lstm
        self.lstm_hidden_size = lstm_hidden_size  # only applicable if b_lstm

        # Initializations (from prior training)
        b_init = torch.tensor(
            [[7.3690e-02, 1.0000e-04, 1.0000e-04, 6.5169e-02, 1.4331e-01,
              2.9631e-03]], dtype=torch.float32
        )
        # k_init = torch.tensor([[.11]], dtype=torch.float32)
        # s_init = torch.tensor([[.20]], dtype=torch.float32)

        # gamma - 5 day (3-7 day) average duration of infection: Woelfel et al
        self.k = Parameter(torch.tensor([[.20]], dtype=torch.float32),
                           requires_grad=False)
        # sigma - 5 day incubation period (	Backer et al )
        self.s = Parameter(torch.tensor([[.20]], dtype=torch.float32),
                           requires_grad=False)

        self.p = Parameter(torch.tensor([[2.5]], dtype=torch.float32),
                           requires_grad=True)
        self.q = Parameter(torch.tensor([[0.2]], dtype=torch.float32),
                           requires_grad=True)

        # Exposed init @ time 0 (no gradient)
        self.e0 = Parameter(torch.tensor(e0, dtype=torch.float32),
                            requires_grad=False)
        # Infected init @ time 0 (no gradient)
        self.i0 = Parameter(torch.tensor(i0, dtype=torch.float32),
                            requires_grad=False)

        # self.p.requires_grad = False

        # if not update_k:  # TODO: disable for now as k is a float...
        #     self.k.requires_grad = False

        if b_lstm:
            print('\nb: Using LSTM\n')
            self.i2l = torch.nn.LSTM(input_size, lstm_hidden_size)
            self.l2b = torch.nn.Linear(lstm_hidden_size, 1)
        else:
            self.i2b = torch.nn.Linear(input_size, 1, bias=False)
            self.i2b.weight.data = b_init

        # Output layer sums I and R (total cases)
        self.h2o = torch.nn.Linear(hidden_size, output_size, bias=False)
        self.h2o.weight.data = torch.from_numpy(
            np.array([1, 1, 0, 0]).astype(np.float32).reshape((1, hidden_size)))
        for p in self.h2o.parameters():
            p.requires_grad = False

    def forward(self, X):
        time_steps = X.size(0)  # time first
        batch_size = X.size(1)  # batch second

        hidden = torch.zeros(
            batch_size, self.hidden_size
        ).to(device=X.device)  # hidden state is i,r,s,e
        hidden[:, 0] = self.i0  # initial infected
        hidden[:, 3] = self.e0  # initial exposed
        hidden[:, 2] = 1.0 - self.i0 - self.e0  # susceptible0
        # hidden[:, 1] = 0.0      # recovered

        p = hidden.clone()  # init previous state
        outputs = []
        hiddens = []
        if self.b_lstm:
            # LSTM states
            h_t = torch.zeros(1, 1, self.lstm_hidden_size)
            c_t = torch.zeros(1, 1, self.lstm_hidden_size)
        for t in range(time_steps):
            # contact rate as a function of our input vector
            if self.b_lstm:
                b_inter, (h_t, c_t) = self.i2l(X[None, t], (h_t, c_t))
                # TODO No negative contact rates...
                # TODO: pytorch does not have LSTM option to change tanh to relu
                #  this is dumb and needs fixing here for valid b...
                b = self.l2b(b_inter.squeeze(dim=1))
                # b = torch.clamp(self.l2b(b_inter.squeeze(dim=1)), 0)
                b = torch.relu(self.l2b(b_inter.squeeze(dim=1)))
                b = b.squeeze()
            else:
                xm = X[t].clone()
                xm[0,5] = 0
                #b = torch.clamp( torch.exp(self.i2b(xm**2)), 0) # predicting the log of the contact rate as a linear combination of mobility squared
                # b = 2.2 # should be the value of b under normal mobility.  Kucharski et al
                # b = 5.0 * .2 * torch.sigmoid(self.i2b(X[t])) # would max out b at 2.2- maybe not a good idea
                b = torch.clamp(self.i2b(xm), 0) ** self.p  # best so far
                # b = 2.2 - 2.2 * torch.tanh(self.i2b(X[t]))  # nope
                # b = 2.2 - 2.2 * torch.tanh(self.i2b(X[t]) ** self.p)  # nope
                # b = self.q * torch.norm(X[t, 0, :5]) ** self.p

            # update the hidden state SIR model (states are I R S E)
            # @formatter:off
            d1 = self.k * p[:, 0]       # gamma * I  (infected people recovering)
            d2 = p[:, 0] * b * p[:, 2]  # b * s * i  (susceptible people becoming exposed)
            d3 = self.s * p[:, 3]       # sigma * e  (exposed people becoming infected)

            hidden[:, 3] = p[:, 3] + d2 - d3  # exposed = exposed + contact_rate * susceptible * infected - sigma * e
            hidden[:, 0] = p[:, 0] + d3 - d1  # infected = infected + s * exposed - infected*recovery_rate
            hidden[:, 1] = p[:, 1] + d1       # recovered = recovered + infected*recovery_rate
            hidden[:, 2] = p[:, 2] - d2       # susceptible
            # @formatter:on

            # update the output
            hidden = torch.clamp(hidden, 0, 1)  # all states must be positive

            p = hidden.clone()
            output = self.h2o(p)
            outputs.append(output)
            hiddens.append(p)

        return torch.stack(hiddens), torch.stack(outputs)
