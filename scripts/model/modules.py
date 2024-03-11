import os
import json
import copy
import math
from collections import OrderedDict
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

from utils.tools import get_mask_from_lengths, pad

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class VarianceAdaptor(nn.Module):
    """ Variance Adaptor """
    def __init__(self, preprocess_config, model_config):
        super(VarianceAdaptor, self).__init__()
        self.duration_predictor = VariancePredictor(model_config)
        self.length_regulator = LengthRegulator()
        self.is_energy, self.is_kurtosis = model_config["variance_embedding"]["is_energy_condition"], model_config["variance_embedding"]["is_kurtosis_condition"]
        if self.is_kurtosis:
            self.kurtosis_predictor = VariancePredictor(model_config)
        if self.is_energy:
            self.energy_predictor = VariancePredictor(model_config)
        kurtosis_quantization = model_config["variance_embedding"]["kurtosis_quantization"]
        energy_quantization = model_config["variance_embedding"]["energy_quantization"]
        n_bins = model_config["variance_embedding"]["n_bins"]
        assert energy_quantization in ["linear", "log"]
        assert kurtosis_quantization in ["linear", "log"]
        with open(
            os.path.join(preprocess_config["path"]["preprocessed"], "stats.json")
        ) as f:
            stats = json.load(f)
            energy_min, energy_max, self.energy_mean, self.energy_std = stats["energy"]
            kurt_min, kurt_max, self.kurtosis_mean, self.kurtosis_std = stats["kurtosis"]
        # kurtosis
        if kurtosis_quantization == "log":
            self.kurt_bins = nn.Parameter(torch.exp(torch.linspace(np.log(kurt_min), np.log(kurt_max), n_bins - 1)),
                requires_grad=False,)
        else:
            self.kurt_bins = nn.Parameter(torch.linspace(kurt_min, kurt_max, n_bins - 1),requires_grad=False,)
        self.kurt_embedding = nn.Embedding(n_bins, model_config["transformer"]["encoder_hidden"])
        # energy
        if energy_quantization == "log":
            self.energy_bins = nn.Parameter(torch.exp(torch.linspace(np.log(energy_min), np.log(energy_max), n_bins - 1)),
                requires_grad=False,)
        else:
            self.energy_bins = nn.Parameter(torch.linspace(energy_min, energy_max, n_bins - 1),requires_grad=False,)
        self.energy_embedding = nn.Embedding(n_bins, model_config["transformer"]["encoder_hidden"])

    def get_energy_embedding(self, x, target, mask, control):
        prediction = self.energy_predictor(x, mask)
        if target is not None:
            embedding = self.energy_embedding(torch.bucketize(target, self.energy_bins))
        else:
            prediction = prediction * self.energy_std + self.energy_mean
            prediction = prediction * control
            prediction = (prediction - self.energy_mean) / self.energy_std
            embedding = self.energy_embedding(
                torch.bucketize(prediction, self.energy_bins)
            )
        return prediction, embedding
    
    def get_kurtosis_embedding(self, x, target, mask, control):
        prediction = self.kurtosis_predictor(x, mask)
        if target is not None:
            embedding = self.kurt_embedding(torch.bucketize(target, self.kurt_bins))
        else:
            prediction = prediction * self.kurtosis_std + self.kurtosis_mean
            prediction = prediction * control
            prediction = (prediction - self.kurtosis_mean) / self.kurtosis_std
            embedding = self.kurt_embedding(
                torch.bucketize(prediction, self.kurt_bins)
            )
        return prediction, embedding
    
    def forward(
        self,
        x,
        src_mask,
        mel_mask=None,
        max_len=None,
        energy_target=None,
        kurtosis_target=None,
        duration_target=None,
        e_control=1.0,
        d_control=1.0,
    ):
        log_duration_prediction = self.duration_predictor(x, src_mask)
        if self.is_energy:
            energy_prediction, energy_embedding = self.get_energy_embedding(
                x, energy_target, src_mask, e_control
            )
            x = x + energy_embedding
        else:
            energy_prediction = None
        if self.is_kurtosis:
            kurtosis_prediction, kurtosis_embedding = self.get_kurtosis_embedding(
                x, kurtosis_target, src_mask, 1.0
            )
            x = x + kurtosis_embedding
        else:
            kurtosis_prediction = None
        if duration_target is not None:
            x, mel_len = self.length_regulator(x, duration_target, max_len)
            duration_rounded = duration_target
        else:
            duration_rounded = torch.clamp(
                (torch.round(torch.exp(log_duration_prediction) - 1) * d_control),
                min=0,
            )
            x, mel_len = self.length_regulator(x, duration_rounded, max_len)
            mel_mask = get_mask_from_lengths(mel_len)
        return (
            x,
            energy_prediction,
            kurtosis_prediction,
            log_duration_prediction,
            duration_rounded,
            mel_len,
            mel_mask,
        )

class LengthRegulator(nn.Module):
    """ Length Regulator """

    def __init__(self):
        super(LengthRegulator, self).__init__()

    def LR(self, x, duration, max_len):
        output = list()
        mel_len = list()
        for batch, expand_target in zip(x, duration):
            expanded = self.expand(batch, expand_target)
            output.append(expanded)
            mel_len.append(expanded.shape[0])

        if max_len is not None:
            output = pad(output, max_len)
        else:
            output = pad(output)

        return output, torch.LongTensor(mel_len).to(device)

    def expand(self, batch, predicted):
        out = list()

        for i, vec in enumerate(batch):
            expand_size = predicted[i].item()
            out.append(vec.expand(max(int(expand_size), 0), -1))
        out = torch.cat(out, 0)

        return out

    def forward(self, x, duration, max_len):
        output, mel_len = self.LR(x, duration, max_len)
        return output, mel_len

class VariancePredictor(nn.Module):

    def __init__(self, model_config):
        super(VariancePredictor, self).__init__()

        self.input_size = model_config["transformer"]["encoder_hidden"]
        self.filter_size = model_config["variance_predictor"]["filter_size"]
        self.kernel = model_config["variance_predictor"]["kernel_size"]
        self.conv_output_size = model_config["variance_predictor"]["filter_size"]
        self.dropout = model_config["variance_predictor"]["dropout"]

        self.conv_layer = nn.Sequential(
            OrderedDict(
                [
                    (
                        "conv1d_1",
                        Conv(
                            self.input_size,
                            self.filter_size,
                            kernel_size=self.kernel,
                            padding=(self.kernel - 1) // 2,
                        ),
                    ),
                    ("relu_1", nn.ReLU()),
                    ("layer_norm_1", nn.LayerNorm(self.filter_size)),
                    ("dropout_1", nn.Dropout(self.dropout)),
                    (
                        "conv1d_2",
                        Conv(
                            self.filter_size,
                            self.filter_size,
                            kernel_size=self.kernel,
                            padding=1,
                        ),
                    ),
                    ("relu_2", nn.ReLU()),
                    ("layer_norm_2", nn.LayerNorm(self.filter_size)),
                    ("dropout_2", nn.Dropout(self.dropout)),
                ]
            )
        )

        self.linear_layer = nn.Linear(self.conv_output_size, 1)

    def forward(self, encoder_output, mask):
        out = self.conv_layer(encoder_output)
        out = self.linear_layer(out)
        out = out.squeeze(-1)

        if mask is not None:
            out = out.masked_fill(mask, 0.0)

        return out


class Conv(nn.Module):
    """
    Convolution Module
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=1,
        stride=1,
        padding=0,
        dilation=1,
        bias=True,
        w_init="linear",
    ):
        """
        :param in_channels: dimension of input
        :param out_channels: dimension of output
        :param kernel_size: size of kernel
        :param stride: size of stride
        :param padding: size of padding
        :param dilation: dilation rate
        :param bias: boolean. if True, bias is included.
        :param w_init: str. weight inits with xavier initialization.
        """
        super(Conv, self).__init__()

        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=bias,
        )

    def forward(self, x):
        x = x.contiguous().transpose(1, 2)
        x = self.conv(x)
        x = x.contiguous().transpose(1, 2)

        return x
