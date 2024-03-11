import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import torchvision

class FastSpeech2Loss(nn.Module):
    """ FastSpeech2 Loss """

    def __init__(self):
        super(FastSpeech2Loss, self).__init__()
        self.mse_loss = nn.MSELoss()
        self.mae_loss = nn.L1Loss()

    def forward(self, inputs, predictions):
        (
            mel_targets,
            mel_lens,
            max_mel_len,
            energy_targets,
            kurtosis_targets,
            duration_targets,
            images,
            event_images
        ) = inputs[5:]
        
        (
            mel_predictions,
            postnet_mel_predictions,
            energy_predictions,
            kurtosis_predictions,
            log_duration_predictions,
            _,
            src_masks,
            mel_masks,
            _,
            _,
        ) = predictions

        src_masks = ~src_masks
        mel_masks = ~mel_masks
        log_duration_targets = torch.log(duration_targets.float() + 1)
        mel_targets = mel_targets[:, : mel_masks.shape[1], :]
        mel_masks = mel_masks[:, :mel_masks.shape[1]]

        log_duration_targets.requires_grad = False
        log_duration_predictions = log_duration_predictions.masked_select(src_masks)
        log_duration_targets = log_duration_targets.masked_select(src_masks)
        mel_targets.requires_grad = False
        mel_predictions = mel_predictions.masked_select(mel_masks.unsqueeze(-1))
        mel_targets = mel_targets.masked_select(mel_masks.unsqueeze(-1))
        mel_loss = self.mae_loss(mel_predictions, mel_targets)

        if energy_targets is not None:
            energy_targets.requires_grad = False
            energy_predictions = energy_predictions.masked_select(src_masks)
            energy_targets = energy_targets.masked_select(src_masks)
            energy_loss = self.mse_loss(energy_predictions, energy_targets)
        else:
            energy_loss = torch.tensor(0).to(mel_targets.device).float()
        if kurtosis_targets is not None:
            kurtosis_targets.requires_grad = False
            kurtosis_predictions = kurtosis_predictions.masked_select(src_masks)
            kurtosis_targets = kurtosis_targets.masked_select(src_masks)
            kurtosis_loss = self.mse_loss(kurtosis_predictions, kurtosis_targets)
        else:
            kurtosis_loss = torch.tensor(0).to(mel_targets.device).float()
        
        
        postnet_mel_predictions = postnet_mel_predictions.masked_select(
            mel_masks.unsqueeze(-1)
        )
        postnet_mel_loss = self.mae_loss(postnet_mel_predictions, mel_targets)
        duration_loss = self.mse_loss(log_duration_predictions, log_duration_targets)

        total_loss = (
            mel_loss + postnet_mel_loss + duration_loss  + energy_loss + kurtosis_loss
        )

        return (
            total_loss,
            mel_loss,
            postnet_mel_loss,
            energy_loss,
            kurtosis_loss,
            duration_loss,
        )
