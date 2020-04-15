import csv
import os
import sys
from collections import OrderedDict

import numpy as np
import matplotlib.pyplot as plt

import argparse

import torch
from torch import optim

ROOT_DIR = os.path.join(os.path.dirname(__file__), '..')
sys.path.append(ROOT_DIR)
import SIRNet


def build_model(b_lstm=False):
    model = torch.nn.Sequential()
    model.add_module("name", SIRNet.SIRNet(b_lstm=b_lstm))
    return model


def train(model, loss, optimizer, x, y):
    optimizer.zero_grad()

    hx, fx = model.forward(x)

    output = loss.forward(fx, y)
    output.backward()
    optimizer.step()
    for p in model.parameters():
        p.data.clamp_(1e-6)
    return output.data.item()


def to_numpy(tensor):
    return tensor.cpu().detach().numpy()


def main(b_lstm=False, no_cuda=False):
    if no_cuda or not torch.cuda.is_available():
        device = torch.device('cpu')  # use CPU
    else:
        device = torch.device('cuda')  # use GPU/CUDA

    ###################### Loading Data #########################
    #############################################################
    # Load Activity Data
    mobility = OrderedDict()
    nyc_data_fn = os.path.join(ROOT_DIR, 'data', 'NYC_data_hacked.csv')
    with open(nyc_data_fn, 'r') as f:
        csv_reader = csv.reader(f, delimiter=',')
        for row in csv_reader:
            date = str(row[1])[1:-1]  # remove the quotes
            mobility[date] = row[2:]  # mobility by date

    # Common Case Data + Activity Data Dates
    data = []
    for k in mobility.keys():
        data.append(mobility[k])

    ###################### Formatting Data ######################
    #############################################################

    data = np.asarray(data).astype(
        float)  # Data is 6 columns of mobility, 1 column of case number
    data = data[5:, :]  # Skip 5 days until we have 10+ patients

    data[:, :6] = (
            1.0 + data[:, :6] / 100.0
    )  # convert percentages of change to fractions of activity

    # Split into input and output data
    X, Y = data[:, :6], data[:, 7]
    # X is now retail&rec, grocery&pharm, parks, transit_stations, workplace, 
    #   residential
    # Y is the total number of cases

    # divide out population of county
    pop = 8.0 * 10 ** 6
    Y = Y / pop
    X = torch.Tensor(X).to(device=device)
    Y = torch.Tensor(Y).to(device=device)

    # Add batch dimension
    X = X.reshape(X.shape[0], 1, X.shape[1])  # time x batch x channels
    Y = Y.reshape(Y.shape[0], 1, 1)  # time x batch x channels

    ##################### Build and Train Model #################
    #############################################################
    model = build_model(b_lstm=b_lstm).to(device=device)
    loss = torch.nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=3000, gamma=0.1)
    batch_size = Y.shape[0]
    torch.autograd.set_detect_anomaly(True)

    if not os.path.exists('weights.pt'):
        for i in range(5000 if b_lstm else 1000):
            cost = 0.
            num_batches = len(X) // batch_size
            for k in range(num_batches):
                start, end = k * batch_size, (k + 1) * batch_size
                cost += train(model, loss, optimizer, X[start:end],
                              Y[start:end])
            if i % 100 == 0:
                print("Epoch = %d, cost = %s" % (i + 1, cost / num_batches))
                print('The model fit is: ')
                for p in model.parameters():
                    print(p.data)
            scheduler.step()

        torch.save(model.state_dict(), 'weights.pt')
    else:
        model.load_state_dict(torch.load('weights.pt'))

    sir_state, total_cases = model(X)
    YY = np.squeeze(np.squeeze(to_numpy(total_cases)))

    # Plot the SIR state
    sir_state = np.squeeze(to_numpy(sir_state))
    plt.plot(sir_state)
    plt.legend(['I', 'R', 'S'])
    plt.xlabel('Day')
    plt.ylabel('Value')
    plt.title('SIR_state')
    plt.show()

    # Plot the total cases
    plt.title('Cases')
    plt.xlabel('Day')
    plt.ylabel('Cases')
    pcs = plt.plot(range(Y.shape[0]), np.squeeze(to_numpy(Y)), 'r',
                   range(Y.shape[0]), YY, 'g')
    plt.legend(pcs, ['Ground Truth', 'Predicted'])
    plt.show()

    ######## Forecast 120 more days at current quarantine mobility #############
    ############################################################################
    xN = X[-1, :, :]
    qX = xN[:, None, ...].expand(120, 1, 6)  # 120 x 1 x 6
    qX = torch.cat((X, qX), axis=0)

    sir_state, total_cases = model(qX)
    YY = np.squeeze(np.squeeze(to_numpy(total_cases)))

    # Plot the SIR state
    sir_state1 = np.squeeze(to_numpy(sir_state))
    plt.plot(sir_state1)
    plt.legend(['I', 'R', 'S'])
    plt.xlabel('Day')
    plt.ylabel('Value')
    plt.title('SIR_state (quarantine mobility)')
    plt.show()

    # Plot the total cases
    plt.title('Cases (quarantine mobility)')
    plt.xlabel('Day')
    plt.ylabel('Cases')
    pcs = plt.plot(range(Y.shape[0]), np.squeeze(to_numpy(Y)), 'r',
                   range(YY.shape[0]), YY, 'g--')
    plt.legend(pcs, ['Ground Truth', 'Forecast'])
    plt.show()

    ######## Forecast 120 more days returning to normal mobility ###############
    ############################################################################
    xN = torch.ones((1, 6), dtype=torch.float32).to(device=device)
    rX = xN[:, None, ...].expand(120, 1, 6)  # 120 x 1 x 6
    rX = torch.cat((X, rX), axis=0)

    sir_state, total_cases = model(rX)
    YY = np.squeeze(np.squeeze(to_numpy(total_cases)))

    # Plot the SIR state
    sir_state2 = np.squeeze(to_numpy(sir_state))
    plt.plot(sir_state2)
    plt.legend(['I', 'R', 'S'])
    plt.xlabel('Day')
    plt.ylabel('Value')
    plt.title('SIR_state (full mobility)')
    plt.show()

    days = range(sir_state1.shape[0])
    pcs = plt.plot(days, sir_state1[:, 0], 'g',
                   days, sir_state2[:, 0], 'r')
    plt.xlabel('Day')
    plt.ylabel('# Infected')
    plt.legend(pcs, ['Current Mobility', 'Normal Mobility'])
    plt.show()

    # Plot the total cases
    plt.title('Cases (full mobility)')
    plt.xlabel('Day')
    plt.ylabel('Cases')
    pcs = plt.plot(range(Y.shape[0]), np.squeeze(to_numpy(Y)), 'r',
                   range(YY.shape[0]), YY, 'g--')
    plt.legend(pcs, ['Ground Truth', 'Forecast'])
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Hybrid SIR & Deep Learning Modeling of COVID-19'
    )
    parser.add_argument('--b-lstm', action='store_true',
                        help='Use an LSTM to learn the `b` parameter of the '
                             'SIR model instead of a single dense layer.')
    parser.add_argument('--no-cuda', action='store_true',
                        help='Disable CUDA (this will automatically be '
                             'disabled if your device does not support CUDA).')
    args = parser.parse_args()
    main(
        b_lstm=args.b_lstm,
        no_cuda=args.no_cuda
    )
