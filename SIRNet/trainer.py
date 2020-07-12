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


class Trainer(object):
    def __init__(self, weights_path):
        # check if weights path exists
        self.weights_path = weights_path

    def build_model(self, e0, i0, b_model='linear', update_k=False):
        # Sequential model
        model = torch.nn.Sequential()
        model.add_module('SEIRNet', SEIRNet(e0=e0, i0=i0, b_model=b_model,
                                            update_k=update_k))
        if os.path.exists(self.weights_path):
            model.load_state_dict(torch.load(self.weights_path),strict=False)
        return model

    def iteration(self, model, loss, optimizer, x, y, log_loss=False):
        optimizer.zero_grad()

        hx, fx = model.forward(x)

        if log_loss:
            output = loss.forward(torch.log(fx), torch.log(y))
        else:
            output = loss.forward(fx, y)
        output.backward()
        optimizer.step()
        # for name, param in model.named_parameters():
        #  if name == "SEIRNet.i2b.weight":
        #     param.data.clamp_(1e-2)
        return output.data.item()

    def train(self, model, X, Y, iters, step_size=4000):
        # Optimizer, scheduler, loss
        loss = torch.nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                    step_size=step_size,
                                                    gamma=0.1)
        torch.autograd.set_detect_anomaly(True)

        for i in range(iters):
            iterator = zip([X], [Y])
            for X, Y in iterator:
                batch_size = Y.shape[0]
                cost = 0.
                num_batches = math.ceil(len(X) / batch_size)
                for k in range(num_batches):
                    start, end = k * batch_size, (k + 1) * batch_size
                    cost += self.iteration(model, loss, optimizer, X[start:end],
                                           Y[start:end])
                if (i + 1) % 50 == 0:
                    print('\nEpoch = %d, cost = %s' %
                          (i + 1, cost / num_batches))
                    print('The model fit is: ')
                    for name, param in model.named_parameters():
                        print(name, param.data)
            # TODO: scheduler may restart learning rate if trying to load from
            #  file. Mitigation: store epoch number in filename
            scheduler.step()
        torch.save(model.state_dict(), self.weights_path)
        return cost  # the final cost
