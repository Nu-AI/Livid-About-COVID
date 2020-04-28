"""The SIRNet package. WIP"""
from .sirnet import SIRNet
from .sirnet import SEIRNet
from . import util
from .trainer import Trainer

__all__ = ['SIRNet', 'SEIRNet', 'util', 'Trainer']
