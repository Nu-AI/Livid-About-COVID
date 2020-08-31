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
from . import metrics
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
            print('Loading weights from file:', self.weights_path)
            model.load_state_dict(torch.load(self.weights_path), strict=False)
        return model

    def iteration(self, model, loss, optimizer, it, x, y, log_loss=False):
        optimizer.zero_grad()

        hx, fx = model.forward(x)

        if self.summary_writer is not None:
            # Note: assuming the SEIRNet module is still being used AND that the
            #  batch size in the hidden states is 1 (squeezed out by to_numpy)
            # states: IRSE
            import matplotlib.pyplot as plt

            hx_np = util.to_numpy(hx)  # squeeze + to numpy array
            # Plot sum over all states (sanity check to ensure sum is always 1)
            f, ax = plt.subplots()
            ax.plot(hx_np.sum(axis=1))
            ax.set_xlabel('day')
            ax.set_ylabel('count')
            ax.ticklabel_format(useOffset=False)
            self.summary_writer.add_figure(self.model_name + '/total',
                                           f, global_step=it)
            plt.close(f)

            f_running, ax_running = plt.subplots()
            ax_running.set_xlabel('day')
            ax_running.set_ylabel('fraction')
            x_running = list(range(len(hx)))
            y2_running = 0
            # Plot the ground truth total cases (infected + recovered)
            ax_running.plot(util.to_numpy(y), c='black')

            for state, name in zip(range(len(hx)),
                                   ['infected', 'recovered', 'susceptible',
                                    'exposed']):
                f, ax = plt.subplots()
                hx_state = hx_np[:, state]
                ax.plot(hx_state)
                ax.set_xlabel('day')
                ax.set_ylabel(name + ' count')
                ylim_l, ylim_h = ax.get_ylim()
                ax.set_ylim(min(ylim_l, 0) - 0.1, max(ylim_h, 1) + 0.1)
                ax.set_yscale('symlog')
                self.summary_writer.add_figure(self.model_name + '/' + name,
                                               f, global_step=it)
                plt.close(f)

                # Running cumulative plot
                y1_running = y2_running + hx_state
                ax_running.fill_between(x_running, y1_running, y2_running)
                y2_running = y1_running
            self.summary_writer.add_figure(self.model_name + '/_IRSE',
                                           f_running, global_step=it)
            plt.close(f_running)

        if log_loss:
            output = loss.forward(torch.log(fx), torch.log(y))
        else:
            output = loss.forward(fx, y)

        output.backward()
        optimizer.step()

        cost = output.data.item()
        return cost

    def train(self, model, X, Y, iters, learning_rate=1e-2, step_size=4000):
        # Optimizer, scheduler, loss
        loss = torch.nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                    step_size=step_size,
                                                    gamma=0.1)
        torch.autograd.set_detect_anomaly(True)

        cost = None
        summary_ignored = set()
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
                        if torch.numel(param) == 1:
                            self.summary_writer.add_scalar(
                                self.model_name + '/' + name, param,
                                global_step=i)
                        elif name not in summary_ignored:
                            print('no summary for', name)
                            summary_ignored.add(name)

            if (i + 1) % 50 == 0 or (i + 1) == iters:
                print('\nEpoch = %d, cost = %s' % (i + 1, cost))
                print('The model model_and_fit is: ')
                for name, param in model.named_parameters():
                    print('   ', name, param.data,
                          'trained={}'.format(param.requires_grad))
                print('MSE={}'.format(self.evaluate(model, X, Y)))

            # TODO: scheduler may restart learning rate if trying to load from
            #  file. Mitigation: store epoch number in filename
            scheduler.step()

        torch.save(model.state_dict(), self.weights_path)

        if self.summary_writer:
            self.summary_writer.add_graph(model, X)

        return cost  # the final cost

    def evaluate(self, model, X, Y):
        # TODO: WIP in function formalization
        YP, _ = model.forward(X)
        mse = metrics.mean_squared_error_samplewise(y_pred=YP, y_true=Y)
        return mse
