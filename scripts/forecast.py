"""
forecast.py - Main file for COVID-19 forecasting.
TODO: Note, this is a WIP file for establishing a unified interface to SIRNet
 training, evaluation, and forecasting. This will become the new centralized
 home for any SIRNet functionality and results, plots, etc.
"""
import os
import sys
import argparse

# Static paths
ROOT_DIR = os.path.join(os.path.dirname(__file__), '..')
sys.path.append(ROOT_DIR)


def main(disable_cuda):
    # Delay expensive import for more rapid argument parsing
    import torch
    import SIRNet

    if disable_cuda or not torch.cuda.is_available():
        device = torch.device('cpu')  # use CPU
    else:
        device = torch.device('cuda')  # use GPU/CUDA


if __name__ == '__main__':
    # The argument parser
    parser = argparse.ArgumentParser(  # noqa
        description='Unified interface to SIRNet training, evaluation, and '
                    'forecasting.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    # Add arguments
    parser.add_argument('--disable-cuda', action='store_true',
                        help='Disables use of CUDA, forcing to execute on CPU '
                             'even if a CPU-capable GPU is available.')
    # Parse arguments
    args = parser.parse_args()

    # Main
    main(
        disable_cuda=args.disable_cuda
    )
