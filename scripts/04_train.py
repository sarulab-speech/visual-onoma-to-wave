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

def _init_logger(train_config):
    for p in train_config["path"].values():
        os.makedirs(p, exist_ok=True)
    os.makedirs(os.path.join(train_config["path"]["result_path"], "Train"), exist_ok=True)
    os.makedirs(os.path.join(train_config["path"]["result_path"], "Val"), exist_ok=True)
    train_log_path = os.path.join(train_config["path"]["log_path"], "train")
    val_log_path = os.path.join(train_config["path"]["log_path"], "val")
    return SummaryWriter(train_log_path), SummaryWriter(val_log_path), train_log_path, val_log_path

def _init_dataset(preprocess_config, train_config):
    dataset = Dataset(
        "train.txt", preprocess_config, train_config, sort=True, drop_last=True
    )
    group_size = 4
    loader = DataLoader(
        dataset,
        batch_size=train_config["optimizer"]["batch_size"] * group_size,
        shuffle=True,
        collate_fn=dataset.collate_fn,
        drop_last=True,
        num_workers=10,
    )
    return dataset, loader

def _init_model(args, configs):
    preprocess_config, model_config, train_config = configs
    model, optimizer = get_model(args, configs, device, train=True)
    Loss = FastSpeech2Loss(preprocess_config, model_config).to(device)
    model = nn.DataParallel(model) if train_config["dataparallel"] else model
    num_param = get_param_num(model)
    vocoder = get_vocoder(model_config, device)
    print("Number of FastSpeech2 Parameters:", num_param)
    return model, optimizer, Loss, vocoder

class YamlConfigManager:
    def __init__(self, configs, loader_len):
        preprocess_config, model_config, train_config = configs
        self.loader_len = loader_len
        self.grad_acc_step = train_config["optimizer"]["grad_acc_step"]
        self.grad_clip_thresh = train_config["optimizer"]["grad_clip_thresh"]
        self.total_step = train_config["step"]["total_step"]
        self.log_step = train_config["step"]["log_step"]
        self.save_step = train_config["step"]["save_step"]
        self.synth_step = train_config["step"]["synth_step"]
        self.val_step = train_config["step"]["val_step"]
        self.outer_bar = tqdm(total=self.total_step, desc="Training", position=0)
        self.inner_bar = tqdm(total=self.loader_len, desc="Epoch 0", position=1)
    
    def inner_bar_refresh(self, epoch):
        self.inner_bar.close()
        self.inner_bar = tqdm(total=self.loader_len, desc=f"Epoch {epoch}", position=1)

def _synth_one_sample(batch, output, configs, use_image, train_logger, step, vocoder):
    preprocess_config, model_config, train_config = configs
    im, wav_reconstruction, wav_prediction, tag = synth_one_sample(
        batch,
        output,
        vocoder,
        model_config,
        preprocess_config,
        use_image,
    )
    log(train_logger, image=im, tag=f"Training/step_{step}_{tag}")
    sampling_rate = preprocess_config["preprocessing"]["audio"]["sampling_rate"]
    write(os.path.join(train_config["path"]["result_path"], "Train", f"{step}_{tag}_reconst.wav"), sampling_rate, wav_reconstruction)
    write(os.path.join(train_config["path"]["result_path"], "Train", f"{step}_{tag}_synthesis.wav"), sampling_rate, wav_prediction)
    log(train_logger, audio=wav_reconstruction, sampling_rate=sampling_rate, tag=f"Training/step_{step}_{tag}_reconst")
    log(train_logger, audio=wav_prediction, sampling_rate=sampling_rate, tag=f"Training/step_{step}_{tag}_synthesis")


def main(args, configs):
    print("Prepare training ...")
    preprocess_config, model_config, train_config = configs
    dataset, loader = _init_dataset(preprocess_config, train_config)
    model, optimizer, Loss, vocoder = _init_model(args, configs)
    train_logger, val_logger, train_log_path, val_log_path = _init_logger(train_config)
    Y = YamlConfigManager(configs, len(loader))
    Y.outer_bar.n = args.restore_step
    Y.outer_bar.update()
    epoch = 1
    step = args.restore_step + 1
    while True:
        Y.inner_bar_refresh(epoch)
        for batchs in loader:
            for batch in batchs:
                # Forward
                batch = to_device(batch, device)
                output = model(*(batch[1:]), train_config["use_image"])
                losses = Loss(batch, output)
                total_loss = losses[0]
                # Backward
                total_loss = total_loss / Y.grad_acc_step
                total_loss.backward()
                if step % Y.grad_acc_step == 0:
                    nn.utils.clip_grad_norm_(model.parameters(), Y.grad_clip_thresh)
                    optimizer.step_and_update_lr()
                    optimizer.zero_grad()         
                # Logging, save checkpoint, run validation
                if step % Y.log_step == 0:
                    losses = [l.item() for l in losses]
                    message1 = "Step {}/{}, ".format(step, Y.total_step)
                    message2 = "Total Loss: {:.4f}, Mel Loss: {:.4f}, Mel PostNet Loss: {:.4f}, Energy Loss: {:.4f}, Duration Loss: {:.4f}".format(*losses)
                    with open(os.path.join(train_log_path, "log.txt"), "a") as f_log:
                        f_log.write(message1 + message2 + "\n")
                    Y.outer_bar.write(message1 + message2)
                    log(train_logger, step, losses=losses)
                if step % Y.synth_step == 0:
                    _synth_one_sample(batch, output, configs, train_config["use_image"], train_logger, step, vocoder)
                if step % Y.val_step == 0:
                    model.eval()
                    message = evaluate(model, step, configs, val_logger, vocoder)
                    with open(os.path.join(val_log_path, "log.txt"), "a") as f_log:
                        f_log.write(message + "\n")
                    Y.outer_bar.write(message)
                    model.train()
                if step % Y.save_step == 0:
                    state_dict = model.module.state_dict() if train_config["dataparallel"] else model.state_dict()
                    torch.save(
                        {
                            "model": state_dict,
                            "optimizer": optimizer._optimizer.state_dict(),
                        },
                        os.path.join(train_config["path"]["ckpt_path"], "{}.pth.tar".format(step)),
                    )
                # check finish
                if step == Y.total_step:
                    quit()
                step += 1
                Y.outer_bar.update(1)
            Y.inner_bar.update(1)
        epoch += 1

if __name__ == "__main__":
    #define args 
    parser = argparse.ArgumentParser()
    parser.add_argument("--restore_step", type=int, default=0)
    parser.add_argument(
        "-p",
        "--preprocess_config",
        type=str,
        required=True,
        help="path to preprocess.yaml",
    )
    parser.add_argument(
        "-m", "--model_config", type=str, required=True, help="path to model.yaml"
    )
    parser.add_argument(
        "-t", "--train_config", type=str, required=True, help="path to train.yaml"
    )
    args = parser.parse_args()

    # Read Config
    preprocess_config = yaml.load(
        open(args.preprocess_config, "r"), Loader=yaml.FullLoader
    )
    model_config = yaml.load(open(args.model_config, "r"), Loader=yaml.FullLoader)
    train_config = yaml.load(open(args.train_config, "r"), Loader=yaml.FullLoader)
    configs = (preprocess_config, model_config, train_config)

    #main function
    main(args, configs)