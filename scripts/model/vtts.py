import os
import json

import torch
from torch import device
import torch.nn as nn
import torch.nn.functional as F

from transformer import Encoder, Decoder, PostNet
from .modules import VarianceAdaptor
from utils.tools import get_mask_from_lengths
from .jdit import JDIT
import clip


class vTTS(nn.Module):
    """ FastSpeech2 """

    def __init__(self, preprocess_config, model_config, train_config):
        super(vTTS, self).__init__()
        self.model_config = model_config
        self.encoder = Encoder(preprocess_config,model_config) # ここから ok: 10:19
        self.variance_adaptor = VarianceAdaptor(preprocess_config, model_config) # 10:21
        self.decoder = Decoder(model_config) # 変更なしのはず
        self.mel_linear = nn.Linear(
            model_config["transformer"]["decoder_hidden"],
            preprocess_config["preprocessing"]["mel"]["n_mel_channels"],
        )
        self.use_jdit = model_config["jdit"]["use_jdit"]
        if self.use_jdit:
            self.jdit = JDIT(model_config=model_config,preprocess_config=preprocess_config)
        self.postnet = PostNet()

        self.audiotype_emb = None
        if model_config["multi_audiotype"]:
            with open(
                os.path.join(
                    preprocess_config["path"]["preprocessed_data_path"], "audiotype.json"
                ),
                "r",
            ) as f:
                n_audiotype = len(json.load(f))
            self.audiotype_emb = nn.Embedding(
                n_audiotype,
                model_config["transformer"]["encoder_hidden"],
            )
        self.image_encoder = train_config["image_encoder"]
        if self.image_encoder:
            device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
            self.linear = nn.Linear(768, 256, bias=False).float().to(device)

    def forward(
        self,
        audiotypes,
        texts,
        src_lens,
        max_src_len,
        mels=None,
        mel_lens=None,
        max_mel_len=None,
        e_targets=None,
        d_targets=None,
        images=None,
        event_image_features=None,
        use_image=True,
        e_control=1.0,
        d_control=1.0
    ):

        src_masks = get_mask_from_lengths(src_lens, max_src_len)
        mel_masks = (
            get_mask_from_lengths(mel_lens, max_mel_len)
            if mel_lens is not None
            else None
        )
        output = self.encoder(texts, src_masks,images=images,use_image=use_image)

        if self.use_jdit:
            assert False
            mel_jdit, gate_outputs, alignments = self.jdit(output, mels, src_lens)

        if self.audiotype_emb is not None:
            if self.image_encoder:
                image_emb = self.linear(event_image_features).expand(
                    -1, max_src_len, -1
                )
                output = output + image_emb
            else:
                output = output + self.audiotype_emb(audiotypes).unsqueeze(1).expand(
                    -1, max_src_len, -1
                )
        if torch.isnan(output).any():
            print("Assert")
        if output.shape[1] != max_src_len:
            print("Assert")
            exit()

        (
            output,
            e_predictions,
            log_d_predictions,
            d_rounded,
            mel_lens,
            mel_masks,
        ) = self.variance_adaptor(
            output,
            src_masks,
            mel_masks,
            max_mel_len,
            e_targets,
            d_targets,
            e_control,
            d_control,
        )

        output, mel_masks = self.decoder(output, mel_masks)
        output = self.mel_linear(output)

        postnet_output = self.postnet(output) + output

        if self.use_jdit:
            assert False
            return (
                output,
                postnet_output,
                p_predictions,
                e_predictions,
                log_d_predictions,
                d_rounded,
                src_masks,
                mel_masks,
                src_lens,
                mel_lens,
                mel_jdit,
                alignments
            )
        else:
            return (
                output,
                postnet_output,
                e_predictions,
                log_d_predictions,
                d_rounded,
                src_masks,
                mel_masks,
                src_lens,
                mel_lens
            )
