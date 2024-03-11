import os
import json
import torch.nn as nn
from transformer import Encoder, Decoder, PostNet
from .modules import VarianceAdaptor
from utils.tools import get_mask_from_lengths
# import clip


class vTTS(nn.Module):
    def __init__(self, preprocess_config, model_config, train_config):
        """ vTTS module
        Args:
            preprocess_config (dict): Preprocess config.
            model_config (dict): Model config.
            train_config (dict): Train config.
        """
        super(vTTS, self).__init__()
        self.model_config = model_config
        self.encoder = Encoder(preprocess_config,model_config)
        self.variance_adaptor = VarianceAdaptor(preprocess_config, model_config)
        self.decoder = Decoder(model_config)
        self.mel_linear = nn.Linear(
            model_config["transformer"]["decoder_hidden"],
            preprocess_config["audio"]["mel"]["n_mel_channels"],
        )
        self.postnet = PostNet()

        self.audiotype_emb = None
        if model_config["multi_audiotype"]:
            with open(
                os.path.join(
                    preprocess_config["path"]["preprocessed"], "audiotype.json"
                ),
                "r",
            ) as f:
                n_audiotype = len(json.load(f))
            self.audiotype_emb = nn.Embedding(
                n_audiotype,
                model_config["transformer"]["encoder_hidden"],
            )
        # self.image_encoder = train_config["image_encoder"]
        # if self.image_encoder:
        #     device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
        #     self.linear = nn.Linear(768, 256, bias=False).float().to(device)

    def forward(
        self, audiotypes, texts, src_lens, max_src_len,
        mels=None, mel_lens=None, max_mel_len=None,
        e_targets=None, k_targets=None, d_targets=None, images=None,
        event_image_features=None, use_image=True,
        e_control=1.0, d_control=1.0
    ):
        """ Forward step
        Args:
            audiotypes (Tensor): Audiotype ids (sound label). (B, )
            texts (Tensor): Text ids. If use_image is True, this is not used. (B, T)
            src_lens (Tensor): Text lengths. (B, )
            max_src_len (int): Max text length.
            mels (Tensor): Mel-spectrogram. (B, mel_channels, T)
            mel_lens (Tensor): Mel lengths. (B, )
            max_mel_len (int): Max mel length.
            e_targets (Tensor): Energy targets. (B, T)
            d_targets (Tensor): Duration targets. (B, T)
            images (Tensor): Image features. (B, C, H, W)
            event_image_features (Tensor): Image features. (B, 768). Now, this is not used.
            use_image (bool): Whether to use image features.
            e_control (float): Energy control. (B, T)
            d_control (float): Duration control. (B, T)
        Returns:
            Tensor: Mel-spectrogram. (B, mel_channels, T)
            Tensor: Postnet output. (B, mel_channels, T)
            Tensor: Energy predictions. (B, T)
            Tensor: Log duration predictions. (B, T)
            Tensor: Duration rounded. (B, T)
            Tensor: Text masks. (B, T)
            Tensor: Mel masks. (B, T)
            Tensor: Text lengths. (B, )
            Tensor: Mel lengths. (B, )
        """
        src_masks = get_mask_from_lengths(src_lens, max_src_len)
        mel_masks = (get_mask_from_lengths(mel_lens, max_mel_len)) if mels is not None else None
        output = self.encoder(texts, src_masks,images=images,use_image=use_image)
        if self.audiotype_emb is not None:
            output = output + self.audiotype_emb(audiotypes).unsqueeze(1).expand(-1, max_src_len, -1)
        (
            output,
            e_predictions,
            k_predictions,
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
            k_targets,
            d_targets,
            e_control,
            d_control,
        )
        output, mel_masks = self.decoder(output, mel_masks)
        output = self.mel_linear(output)
        postnet_output = self.postnet(output) + output
        return (
            output,
            postnet_output,
            e_predictions,
            k_predictions,
            log_d_predictions,
            d_rounded,
            src_masks,
            mel_masks,
            src_lens,
            mel_lens
        )
