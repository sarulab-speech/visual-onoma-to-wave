from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont

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
        self.p_acceptance_score_border = config_dict["preprocessing"]["acceptance_score_border"]

        self.aug_maxlen = config_dict["preprocessing"]["augmentation"]["augment_maxlen"]
        self.aug_repeatnum = config_dict["preprocessing"]["augmentation"]["augment_repeatnum"]
        self.aug_chara_consecutive_num = config_dict["preprocessing"]["augmentation"]["chara_consecutive_num"]
        self.aug_first_consecutive = config_dict["preprocessing"]["augmentation"]["first_consecutive"]

        self.im_fontsize = config_dict["preprocessing"]["text"]["font_size"]
        self.im_is_stretching = config_dict["preprocessing"]["image"]["image_stretching"]
        self.im_bgcolor = tuple(config_dict["preprocessing"]["image"]["background_color"])
        self.im_txtcolor = tuple(config_dict["preprocessing"]["image"]["text_color"])
        self.im_padcolor = config_dict["preprocessing"]["image"]["pad_color"]
        self.im_loadscale = config_dict["preprocessing"]["image"]["load_scale"]

        self.sampling_rate = config_dict["preprocessing"]["audio"]["sampling_rate"]
        self.max_wav_value = config_dict["preprocessing"]["audio"]["max_wav_value"]
        self.filter_length = config_dict["preprocessing"]["stft"]["filter_length"]
        self.hop_length = config_dict["preprocessing"]["stft"]["hop_length"]
        self.win_length = config_dict["preprocessing"]["stft"]["win_length"]
        self.margin_frame = config_dict["preprocessing"]["stft"]["margin_frame"]
        self.n_mel_channels = config_dict["preprocessing"]["mel"]["n_mel_channels"]
        self.mel_fmin = config_dict["preprocessing"]["mel"]["mel_fmin"]
        self.mel_fmax = config_dict["preprocessing"]["mel"]["mel_fmax"]

        self.energy_feature = config_dict["preprocessing"]["energy"]["feature"]
        self.energy_character_averaging = (
            self.energy_feature == "element_level"
        )
        self.energy_normalization = config_dict["preprocessing"]["energy"]["normalization"]

    def __init__(self, config, chara_persec, max_width):
        self._config_open(config)
        self.chara_persec = chara_persec
        self.max_width = max_width

    def _canvas_allocate(self, text_len, canvas_width):
        """ https://qiita.com/keisuke-nakata/items/c18cda4ded06d3159109 """
        return np.array([(canvas_width + i)//text_len for i in range(text_len)]).astype(np.int32)

    def draw(self, text, wav_sec=None, save_im=None, save_width=None):
        font = ImageFont.truetype(str(self.path_font), self.im_fontsize)
        if self.im_is_stretching:
            assert wav_sec is not None, "In stretching mode, wav_sec must be specified."
            canvas_width = np.ceil(self.chara_persec * wav_sec * self.im_fontsize)
        else:
            canvas_width = self.im_fontsize * len(text)
        canvas_height = self.im_fontsize
        canvas_width = canvas_width.astype(np.int32)
        canvas = Image.new("RGB", (canvas_width, canvas_height), self.im_bgcolor)
        character_widths = self._canvas_allocate(len(text), canvas_width)
        w = 0
        for char, width in zip(text, character_widths):
            c_im = Image.new("RGB", (self.im_fontsize, self.im_fontsize), self.im_bgcolor)
            draw = ImageDraw.Draw(c_im)
            draw.text((0, 0), char, fill=self.im_txtcolor, font=font)
            c_im = c_im.resize((width, self.im_fontsize)) if self.im_is_stretching else c_im
            canvas.paste(c_im, (w, 0))
            w += width
        if save_im is not None:
            canvas.save(save_im)
        if save_width is not None:
            np.save(save_width, character_widths)
        return canvas, character_widths
