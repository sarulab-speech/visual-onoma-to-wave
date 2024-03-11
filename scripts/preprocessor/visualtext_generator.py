from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont

class Generator:
    def _config_open(self, config_dict):
        self.path_font = Path(config_dict["path"]["font"])
        self.im_fontsize = config_dict["visual_text"]["fontsize"]
        self.im_is_stretching = config_dict["visual_text"]["image_stretching"]
        self.im_bgcolor = tuple(config_dict["visual_text"]["color"]["background"])
        self.im_txtcolor = tuple(config_dict["visual_text"]["color"]["text"])
        self.im_loadscale = config_dict["visual_text"]["scale_in_training"]

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
