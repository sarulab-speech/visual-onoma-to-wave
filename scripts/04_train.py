import argparse
import os

import torch
from torch.nn.parallel.data_parallel import data_parallel
import yaml
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from utils.model import get_model, get_vocoder, get_param_num
from utils.tools import to_device, log, synth_one_sample, plot_alignment_to_numpy
from model import FastSpeech2Loss
from dataset import Dataset
from scipy.io.wavfile import write
from evaluate import evaluate
import audio as Audio

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")