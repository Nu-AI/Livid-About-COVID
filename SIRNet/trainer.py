"""
Trainer class usage:

```
trainer = Trainer(weights_path)
model = trainer.build_model(e0, i0)
trainer.train(model, X, Y, iters)
```
"""
import os
import math

import torch

from .sirnet import SEIRNet
from . import util


class Trainer(object):
    def __init__(self, weights_path, summary_writer=None):
        self.weights_path = weights_path  # to save/restore weights
        self.summary_writer = summary_writer  # for writing summaries
        self.model_name = 'SEIRNet'

    def build_model(self, e0, i0, b_model='linear', update_k=False):
        # Sequential model
        model = torch.nn.Sequential()
        model.add_module(self.model_name, SEIRNet(
            e0=e0, i0=i0, b_model=b_model, update_k=update_k,
            summary_writer=self.summary_writer
        ))
        if os.path.exists(self.weights_path):
            model.load_state_dict(torch.load(self.weights_path), strict=False)
        return model

    def iteration(self, model, loss, optimizer, it, x, y, log_loss=False):
        optimizer.zero_grad()

        hx, fx = model.forward(x)

        if self.summary_writer is not None:
            # Note: assuming the SEIRNet module is still being used
            # states: IRSE
            import matplotlib.pyplot as plt

            for state, name in zip(range(len(hx)),
                                   ['infected', 'recovered', 'susceptible',
                                    'exposed']):
                f, ax = plt.subplots()
                ax.plot(util.to_numpy(hx)[:, state])
                ax.set_xlabel('day')
                ax.set_ylabel(name + ' count')
                self.summary_writer.add_figure(self.model_name + '/' + name,
                                               f, global_step=it)
                plt.close(f)

        if log_loss:
            output = loss.forward(torch.log(fx), torch.log(y))
        else:
            output = loss.forward(fx, y)

        output.backward()
        optimizer.step()

        cost = output.data.item()
        return cost

    def train(self, model, X, Y, iters, step_size=4000):
        # Optimizer, scheduler, loss
        loss = torch.nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                    step_size=step_size,
                                                    gamma=0.1)
        torch.autograd.set_detect_anomaly(True)

        cost = None
        for i in range(iters):
            batch_size = X.shape[1]  # TODO
            cost = 0.
            num_batches = math.ceil(X.shape[1] / batch_size)
            for k in range(num_batches):
                start, end = k * batch_size, (k + 1) * batch_size
                cost += self.iteration(model, loss, optimizer,
                                       i * batch_size + k,
                                       X[:, start:end], Y[:, start:end])
            cost /= num_batches
            if self.summary_writer:
                self.summary_writer.add_scalar(self.model_name + '/cost',
                                               cost, global_step=i)
                for name, param in model.named_parameters():
                    if param.requires_grad:
                        self.summary_writer.add_scalar(
                            self.model_name + '/' + name, param, global_step=i)

            if (i + 1) % 50 == 0 or (i + 1) == iters:
                print('\nEpoch = %d, cost = %s' % (i + 1, cost))
                print('The model model_and_fit is: ')
                for name, param in model.named_parameters():
                    print('   ', name, param.data,
                          'trained={}'.format(param.requires_grad))

            # TODO: scheduler may restart learning rate if trying to load from
            #  file. Mitigation: store epoch number in filename
            scheduler.step()

        torch.save(model.state_dict(), self.weights_path)

        if self.summary_writer:
            self.summary_writer.add_graph(model, X)

        return cost  # the final cost
