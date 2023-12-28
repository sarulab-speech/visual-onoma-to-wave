import os
import json

from cv2 import normalize
import torch
import numpy as np
import hifigan
from model import vTTS, ScheduledOptim

def get_model(restore_step, configs, device, train=False):
    (preprocess_config, model_config, train_config) = configs
    model = vTTS(preprocess_config, model_config, train_config).to(device)
    if restore_step:
        print("restore_step",restore_step)
        ckpt_path = os.path.join(
            train_config["path"]["ckpt_path"],
            "{}.pth.tar".format(restore_step),
        )
        ckpt = torch.load(ckpt_path)
        model.load_state_dict(ckpt["model"])

    if train:
        scheduled_optim = ScheduledOptim(
            model, train_config, model_config, restore_step
        )
        if restore_step:
            scheduled_optim.load_state_dict(ckpt["optimizer"])
        model.train()
        return model, scheduled_optim

    model.eval()
    model.requires_grad_ = False
    return model


def get_param_num(model):
    num_param = sum(param.numel() for param in model.parameters())
    return num_param


def get_vocoder(config, device):
    name = config["vocoder"]["model"]
    speaker = config["vocoder"]["speaker"]

    if name == "MelGAN":
        if speaker == "LJSpeech":
            vocoder = torch.hub.load(
                "descriptinc/melgan-neurips", "load_melgan", "linda_johnson"
            )
        elif speaker == "universal":
            vocoder = torch.hub.load(
                "descriptinc/melgan-neurips", "load_melgan", "multi_speaker"
            )
        vocoder.mel2wav.eval()
        vocoder.mel2wav.to(device)
    elif name == "HiFi-GAN":
        with open("scripts/hifigan/config.json", "r") as f:
            config = json.load(f)
        config = hifigan.AttrDict(config)
        vocoder = hifigan.Generator(config)
        if speaker == "LJSpeech":
            ckpt = torch.load("scripts/hifigan/generator_LJSpeech.pth.tar")
        elif speaker == "universal":
            ckpt = torch.load("scripts/hifigan/generator_universal.pth.tar")
        vocoder.load_state_dict(ckpt["generator"])
        vocoder.eval()
        vocoder.remove_weight_norm()
        vocoder.to(device)

    return vocoder


def vocoder_infer(mels, vocoder, model_config, preprocess_config, lengths=None, Normalize=True):
    name = model_config["vocoder"]["model"]
    with torch.no_grad():
        if name == "MelGAN":
            wavs = vocoder.inverse(mels / np.log(10))
        elif name == "HiFi-GAN":
            wavs = vocoder(mels).squeeze(1)

    a = torch.max(wavs[0])
    b = torch.min(wavs[0])
    if Normalize:
        wavs = (
            wavs.cpu().numpy()
            * preprocess_config["preprocessing"]["audio"]["max_wav_value"]
        ).astype("int16")
    else:
        wavs = (
            wavs.cpu().numpy()
        )
    wavs = [wav for wav in wavs]

    for i in range(len(mels)):
        if lengths is not None:
            wavs[i] = wavs[i][: lengths[i]]

    return wavs
