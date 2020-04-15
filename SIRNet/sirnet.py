import numpy as np
import torch
from torch.autograd import Variable
from torch.nn.parameter import Parameter


###################### Defining Model #######################
#############################################################
class SIRNet(torch.nn.Module):
    def __init__(self, input_size=6, i0=5.6e-6, update_k=True, hidden_size=3, output_size=1,
                 b_lstm=False):
        super(SIRNet, self).__init__()

        assert input_size == 6, 'Input dimension must be 6'  # for now
        assert hidden_size == 3, 'Hidden dimension must be 3'  # for now
        assert output_size == 1, 'Output dimension must be 1'  # for now

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.b_lstm = b_lstm

        # Initializations (from prior training)
        b_init = torch.from_numpy(np.asarray(
            [1.2276e-01, 1.0000e-06, 1.0000e-06,
             1.1405e-01, 1.9223e-01, 5.0087e-02]
        ).astype(np.float32)).reshape((1, input_size))
        k_init = torch.from_numpy(
            np.asarray([.0898]).astype(np.float32)).reshape((1, 1))

        self.k = Parameter(k_init) if update_k else k_init
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
        hidden = Variable(
            torch.zeros(batch_size, self.hidden_size)
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
                b = self.i2b(X[t])

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
