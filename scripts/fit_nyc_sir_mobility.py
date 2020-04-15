import numpy as np
import csv
import os
from collections import OrderedDict

###################### Defining Model #######################
#############################################################
import torch
from torch.autograd import Variable
from torch import optim
from torch.nn.parameter import Parameter


class TestModel(torch.nn.Module):

    def __init__(self):
        super(TestModel, self).__init__()

        # Initializations (from prior training)
        b_init = torch.from_numpy(
            np.array([0.1633, -.1748, 0.0401, 0.1545, 0.2326, 0.0898]).astype(
                np.float32)).reshape((6, 1))
        k_init = torch.from_numpy(np.array([.0898]).astype(np.float32)).reshape(
            (1, 1))

        self.b = Parameter(b_init)
        self.k = Parameter(k_init)
        # initial population (leave constant/consider solved because it's
        #  touchy)
        self.i0 = 5.6e-6  # 5e-7

    def forward(self, x):
        # Contact rate as a function of time
        # Nx6 * 6x1  - force positive coefficients
        contact_rate = torch.mm(x[:, :6], self.b).reshape((1, -1))

        infected = torch.zeros_like(contact_rate)
        recovered = torch.zeros_like(contact_rate)
        susceptible = torch.zeros_like(contact_rate)
        infected[0, 0] = self.i0  # one person
        susceptible[0, 0] = 1.0 - self.i0
        for t in range(1, infected.shape[1]):
            infected[0, t] = (
                    infected[0, t - 1] + infected.clone()[0, t - 1] *
                    contact_rate[0, t - 1] * susceptible.clone()[0, t - 1] -
                    torch.abs(self.k) * infected.clone()[0, t - 1]
            )  # Force positive recover rate
            recovered[0, t] = (
                    recovered.clone()[0, t - 1] +
                    torch.abs(self.k) * infected.clone()[0, t - 1]
            )
            susceptible[0, t] = (
                    1.0 - recovered.clone()[0, t] - infected.clone()[0, t]
            )

        # A silly hack to print this sometimes
        if np.random.uniform() > 0.99:
            print(infected + recovered)
            print(self.b, self.k)
        return infected + recovered


def build_model():
    model = torch.nn.Sequential()
    model.add_module("name", TestModel())
    return model


def train(model, loss, optimizer, x, y):
    x = Variable(x, requires_grad=False)
    y = Variable(y, requires_grad=False)
    optimizer.zero_grad()

    fx = model.forward(x)
    output = loss.forward(fx, y)
    output.backward()
    optimizer.step()
    return output.data.item()


def main():
    ###################### Loading Data #########################
    #############################################################

    # Load Activity Data
    mobility = OrderedDict()
    # CSV is in the data directory
    nyc_data_fn = os.path.join(os.path.dirname(__file__), '..',
                               'data', 'NYC_data_hacked.csv')
    with open(nyc_data_fn, 'r') as f:
        csv_reader = csv.reader(f, delimiter=',')
        # header = next(csv_reader)
        for row in csv_reader:
            date = str(row[1])[1:-1]  # remove the quotes
            mobility[date] = row[2:]  # mobility by date

    # Common Case Data + Activity Data Dates
    data = []

    for k in mobility.keys():
        data.append(mobility[k])

    ###################### Formatting Data ######################
    #############################################################

    # Data is 6 columns of mobility, 1 column of case number
    data = np.array(data).astype(float)
    data = data[5:, :]
    print(data)

    # convert percentages change to fraction of activity
    data[:, :6] = 1.0 + data[:, :6] / 100.0

    # Split into input and output data
    X, Y = data[:, :6], data[:, 7]

    # X is now retail&rec, grocery&pharm, parks, transit_stations, workplace,
    #  residential

    # divide out population of county
    pop = 8.0 * 10 ** 6
    Y = Y / pop
    X = torch.Tensor(X)
    Y = torch.Tensor(Y)
    print(X.shape, Y.shape)

    ##################### Build and Train Model #################
    #############################################################

    model = build_model()
    loss = torch.nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=3000, gamma=0.1)
    batch_size = Y.shape[0]
    torch.autograd.set_detect_anomaly(True)

    print('The initial fit is: ')
    for p in model.parameters():
        print(p.data)

    for i in range(1000):
        cost = 0.
        num_batches = len(X) // batch_size
        for k in range(num_batches):
            start, end = k * batch_size, (k + 1) * batch_size
            cost += train(model, loss, optimizer, X[start:end], Y[start:end])
        if i % 100 == 0:
            print("Epoch = %d, cost = %s" % (i + 1, cost / num_batches))
        scheduler.step()

    # print(header[2:])
    print('The model fit is: ')
    for p in model.parameters():
        print(p.data)

    import matplotlib.pyplot as plt

    YY = np.squeeze(np.array(model(Variable(X)).data))

    plt.title('Cases')
    plt.xlabel('Day')
    plt.ylabel('Cases')
    pcs = plt.plot(range(Y.shape[0]), np.squeeze(np.array(Y)), 'r',
                   range(Y.shape[0]), YY, 'g')
    plt.legend(pcs, ['Ground Truth', 'Predicted'])
    plt.show()


if __name__ == '__main__':
    main()
