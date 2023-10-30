from pathlib import Path
from tqdm import tqdm
import random
import librosa
import numpy as np
import os
import json
import tgt
from sklearn.preprocessing import StandardScaler
import torchaudio
import torch
import joblib

class Generator:

    def _config_open(self, config_dict):
        self.path_corpus = Path(config_dict["path"]["corpus_data_path"])
        self.path_formatted = Path(config_dict["path"]["formatted_data_path"])
        self.path_preprocessed = Path(config_dict["path"]["preprocessed_data_path"])
        self.path_font = Path(config_dict["path"]["font_path"])

        self.p_uselabel = config_dict["preprocessing"]["extract_label"]
        self.p_untrained_dataid = config_dict["preprocessing"]["notuse_train_audio_num"]
        self.p_inputtype = config_dict["preprocessing"]["input_type"]
        self.p_confidence_score_border = config_dict["preprocessing"]["confidence_score_border"]
        self.p_accentance_score_border = config_dict["preprocessing"]["accentance_score_border"]

        self.aug_maxlen = config_dict["augmentation"]["augment_maxlen"]
        self.aug_repeatnum = config_dict["augmentation"]["augment_repeatnum"]
        self.aug_chara_consecutive_num = config_dict["augmentation"]["chara_consecutive_num"]
        self.aug_first_consecutive = config_dict["augmentation"]["first_consecutive"]

        self.im_fontsize = config_dict["text"]["font_size"]
        self.im_is_stretching = config_dict["image"]["image_stretching"]
        self.im_bgcolor = config_dict["image"]["background_color"]
        self.im_txtcolor = config_dict["image"]["text_color"]
        self.im_padcolor = config_dict["image"]["pad_color"]
        self.im_loadsclae = config_dict["image"]["load_scale"]

        self.sampling_rate = config_dict["audio"]["sampling_rate"]
        self.max_wav_value = config_dict["audio"]["max_wav_value"]
        self.filter_length = config_dict["stft"]["filter_length"]
        self.hop_length = config_dict["stft"]["hop_length"]
        self.win_length = config_dict["stft"]["win_length"]
        self.margin_frame = config_dict["stft"]["margin_frame"]
        self.n_mel_channels = config_dict["mel"]["n_mel_channels"]
        self.mel_fmin = config_dict["mel"]["mel_fmin"]
        self.mel_fmax = config_dict["mel"]["mel_fmax"]

        self.energy_feature = config_dict["energy"]["feature"]
        self.energy_character_averaging = (
            self.energy_feature == "element_level"
        )
        self.energy_normalization = config_dict["energy"]["normalization"]

    def __init__(self, config):
        self._config_open(config)
        if self.im_is_stretching:
            self.charanum_1sec, self.max_width = self.compute_width()
        else:
            self.max_width = self.im_fontsize