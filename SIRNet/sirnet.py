from abc import ABC, abstractmethod

import torch
from torch.nn.parameter import Parameter


class SIRNetBase(ABC, torch.nn.Module):
    # Subclasses must set this. For SIR, the 3 compartments would be S, I, R...
    N_MOBILITY = 6
    N_COMPARTMENTS = 3
    N_OUTPUTS = 1

    def __init__(self, input_size=6, hidden_size=3, output_size=1, i0=5.6e-6,
                 k=0.2, b_model='linear', update_k=False, b_kwargs=None,
                 summary_writer=None):
        # TODO(document): b_kwargs are passed to _make_b_model and are not
        #  standard **kwargs as those may be used for other attributes or
        #  sub-models
        super(SIRNetBase, self).__init__()

        # Number of mobility points
        # TODO: this should be able to be reduced at least
        assert input_size == self.N_MOBILITY, \
            'Input dimension must be %d' % self.N_MOBILITY
        # Number of compartments of the SIR-like model
        assert hidden_size == self.N_COMPARTMENTS, \
            'Hidden dimension must be %d' % self.N_COMPARTMENTS
        # Number of cases or other outputs
        assert output_size == self.N_OUTPUTS, \
            'Output dimension must be %d' % self.N_OUTPUTS
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        # gamma=.2 : 5 day (3-7 day) average duration of infection [Woelfel
        # et al]
        self.k = Parameter(torch.tensor([[k]], dtype=torch.float32),
                           requires_grad=update_k)
        self.sd = Parameter(torch.tensor([[0.5]], dtype=torch.float32),
                            requires_grad=True)
        # b
        self.b_model = b_model.lower()
        self._make_b_model(**(b_kwargs or {}))
        # i0 - infected init @ time 0 (no gradient)
        self.i0 = Parameter(torch.tensor(i0, dtype=torch.float32),
                            requires_grad=False)

        self.summary_writer = summary_writer

    def _make_b_model(self, lstm_hidden_size=None, lstm_bias=False):
        """Transforms input to b parameter of SIR-like models"""
        self.norm_of_mobility = True  # TODO

        if self.b_model == 'linear':
            # Initialization from prior training
            b_init = torch.tensor(
                [[7.3690e-02, 1.0000e-04, 1.0000e-04, 6.5169e-02, 1.4331e-01,
                  2.9631e-03]], dtype=torch.float32
            )
            self.i2b = torch.nn.Linear(self.input_size, 1, bias=False)
            self.i2b.weight.data = b_init

            # p and q
            self.p = Parameter(torch.tensor([[2.5]], dtype=torch.float32),
                               requires_grad=True)
            self.q = Parameter(torch.tensor([[0.2]], dtype=torch.float32),
                               requires_grad=True)

        elif self.b_model == 'lstm':
            if lstm_hidden_size is None:
                lstm_hidden_size = self.N_MOBILITY  # good default
            input_size = 1 if self.norm_of_mobility else self.input_size
            self.i2l = torch.nn.LSTM(input_size, lstm_hidden_size,
                                     bias=lstm_bias)
            # self.l2b = torch.nn.Linear(lstm_hidden_size, 1)
            # TODO
            self.l2b = torch.nn.Linear(1, 1)
        else:
            raise ValueError('b_model must be either "linear" or "lstm" but '
                             'received: {}'.format(self.b_model))

    @abstractmethod
    def _forward_init(self, hidden):
        """Subclasses must implement this to properly initialize hidden and call
        this via super() to initialize the b_model properly"""
        if self.b_model == 'lstm':
            # LSTM states
            self.h_t = torch.zeros(1, 1, self.i2l.hidden_size)
            self.c_t = torch.zeros(1, 1, self.i2l.hidden_size)

    def _forward_b(self, xt):
        """Contact rate as a function of our input vector"""
        if self.b_model == 'lstm':
            if self.norm_of_mobility:
                xt_ = torch.norm(xt).reshape(1, 1, 1)
            else:
                xt_ = xt[None, ...]
            b_inter, (self.h_t, self.c_t) = self.i2l(xt_, (self.h_t, self.c_t))
            b_inter = b_inter.squeeze(dim=1)
            # TODO No negative contact rates...pytorch does not have LSTM
            #  option to change tanh to relu - needs custom LSTM implementation
            #  in Python to change activation function to mitigate negatives...

            # Multiply by empirical scalar...
            # b = torch.relu(self.l2b(b_inter)).squeeze()  # * 5.
            b = torch.relu(self.l2b(
                torch.norm(b_inter).reshape(1, 1))
            ).squeeze()  # * 5.
        elif self.b_model == 'linear':
            # Just look at norm of mobility
            # xm = xt.clone()
            # b = ((1 - torch.sigmoid(self.sd) * xt[0, 5]) *
            #      self.q * torch.norm(xm) ** self.p)

            b = self.q * torch.norm(xt)
            # b = self.q * torch.norm(xt) ** torch.log(self.p)

            # b = torch.relu(self.i2b(xt)) ** self.p
        else:
            raise RuntimeError('b_model is invalid, this should not have '
                               'happened')  # earlier check in _make_b_model()
        return b

    @abstractmethod
    def _forward_update_state(self, hidden, prev_h, b, *args, **kwargs):
        """Update the SIR-like state of the model. Subclasses must implement
        this. Subclasses should return an updated version of `hidden`"""
        raise NotImplementedError

    def _forward_output(self, hidden):
        """It may be a good idea to update this in subclasses, and necessarily
        must do so is the order of the SIR-like compartments does NOT begin with
        I, R"""
        return (hidden[:, 0] + hidden[:, 1])[..., None]  # add dimension w/ None

    def _forward_cleanup(self):
        if self.b_model == 'lstm':
            # Remove LSTM state storage
            del self.h_t
            del self.c_t

    def forward(self, X):
        time_steps = X.size(0)  # time first
        batch_size = X.size(1)  # batch second
        hidden = torch.zeros(
            batch_size, self.hidden_size
        ).to(device=X.device)  # hidden state is i,r,s,...
        # Initialize for the forward pass
        self._forward_init(hidden)
        # TODO: include init hidden in output hiddens?
        prev_h = hidden.clone()  # init previous state
        hiddens = []
        outputs = []
        for t in range(time_steps):
            # compute b
            b = self._forward_b(X[t])
            if self.summary_writer is not None:
                # TODO: note: this will only be helpful for a single region, for
                #  a single summary writer, for a single forecast of mobility
                #  cases, etc.
                self.summary_writer.add_scalar(
                    self.__class__.__name__ + '/b',
                    b, global_step=t
                )
            # update the hidden state of SIR-like model
            hidden = self._forward_update_state(hidden, prev_h, b)
            # update the outputs
            prev_h = hidden.clone()
            outputs.append(self._forward_output(prev_h))
            hiddens.append(prev_h)
        # End of loop cleanup
        self._forward_cleanup()

        return torch.stack(hiddens), torch.stack(outputs)


class SIRNet(SIRNetBase):
    N_COMPARTMENTS = 3
    N_OUTPUTS = 1

    def _forward_init(self, hidden):
        hidden[:, 0] = self.i0
        hidden[:, 2] = 1.0 - self.i0
        super()._forward_init(hidden)

    def _forward_update_state(self, hidden, prev_h, b):  # noqa
        # update the hidden state SIR model @formatter:off
        drdt = self.k * prev_h[:, 0]
        hidden[:, 0] = prev_h[:, 0] + prev_h[:, 0] * b * prev_h[:, 2] - drdt  # infected
        hidden[:, 1] = prev_h[:, 1] + drdt                                    # recovered
        hidden[:, 2] = 1.0 - prev_h[:, 0] - prev_h[:, 1]                      # susceptible
        # @formatter:on
        return hidden


class SEIRNet(SIRNetBase):
    N_COMPARTMENTS = 4
    N_OUTPUTS = 1

    def __init__(self, *args, hidden_size=4, e0=5.6e-6, update_s=False,
                 **kwargs):
        super(SEIRNet, self).__init__(*args, hidden_size=hidden_size, **kwargs)

        # sigma - 5 day incubation period [Backer et al]
        self.s = Parameter(torch.tensor([[.20]], dtype=torch.float32),
                           requires_grad=update_s)
        # Exposed init @ time 0 (no gradient)
        self.e0 = Parameter(torch.tensor(e0, dtype=torch.float32),
                            requires_grad=False)

    def _forward_init(self, hidden):
        hidden[:, 0] = self.i0  # initial infected
        hidden[:, 2] = 1.0 - self.i0 - self.e0  # susceptible
        hidden[:, 3] = self.e0  # initial exposed
        super()._forward_init(hidden)

    def _forward_update_state(self, hidden, prev_h, b):  # noqa
        # update the hidden state SIR model (states are I R S E)
        # @formatter:off
        d1 = self.k * prev_h[:, 0]             # gamma * I  (infected people recovering)
        d2 = prev_h[:, 0] * b * prev_h[:, 2]   # b * s * i  (susceptible people becoming exposed)
        d3 = self.s * prev_h[:, 3]             # sigma * e  (exposed people becoming infected)

        hidden[:, 3] = prev_h[:, 3] + d2 - d3  # exposed = exposed + contact_rate * susceptible * infected - sigma * e
        hidden[:, 0] = prev_h[:, 0] + d3 - d1  # infected = infected + s * exposed - infected*recovery_rate
        hidden[:, 1] = prev_h[:, 1] + d1       # recovered = recovered + infected*recovery_rate
        hidden[:, 2] = prev_h[:, 2] - d2       # susceptible
        # @formatter:on
        # TODO: this won't sum up correctly, should we min-max normalize
        #  instead?
        return torch.clamp(hidden, 0, 1)  # all states must be positive
