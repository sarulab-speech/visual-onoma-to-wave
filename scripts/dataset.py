import json
import os
import cv2
import numpy as np
from torch.utils.data import Dataset
import PIL
from tqdm import tqdm
from pathlib import Path

from utils.symbols import get_symbols
from utils.tools import pad_1D, pad_2D,pad_2D_image,pad_2D_gray_image

class Dataset(Dataset):
    def __init__(
        self, filename, preprocess_config, train_config, model_config, sort=False, drop_last=False
    ):
        """ Dataset for training
        Args:
            preprocess_config (dict): Preprocess config.
            train_config (dict): Train config.
            model_config (dict): Model config.
            filename (str): Filename for training.
            sort (bool): Sort data by text length.
            drop_last (bool): Drop last batch.
        """
        #basic info
        self.preprocessed_path = Path(preprocess_config["path"]["preprocessed"])
        self.batch_size = train_config["optimizer"]["batch_size"]
        self.input_type = preprocess_config["input_type"]
        self.symbol_to_id = get_symbols(self.preprocessed_path)
        self.sort = sort
        self.drop_last = drop_last
        self.use_image = train_config["use_image"]
        self.is_energy = model_config["variance_embedding"]["is_energy_condition"]
        self.is_kurtosis = model_config["variance_embedding"]["is_kurtosis_condition"]

        if self.input_type == 'visual-text':
            #image information
            self.text_font_size = preprocess_config["visual_text"]["fontsize"]
            self.image_bgcolor = preprocess_config["visual_text"]["color"]["background"]
            self.image_textcolor = preprocess_config["visual_text"]["color"]["text"]
            self.image_loadscale = preprocess_config["visual_text"]["scale_in_training"]
            with open(self.preprocessed_path/"visual_text.json") as f:
                visual_text_info = json.load(f)
            self.width = visual_text_info["max_pixelsize"][0]
            self.height = visual_text_info["height"][0]
            self.stride = preprocess_config["visual_text"]["stride"]
            self.basename, self.audiotype, self.fontsize, self.fonttype, self.text = self.process_meta(filename)

        with open(self.preprocessed_path/"audiotype.json") as f:
            self.audiotype_map = json.load(f)
        # self.use_image_encoder = train_config["image_encoder"]
        # if self.use_image_encoder:
        #     self.event_image_path = preprocess_config["path"]["event_image_path"]
        #     self.event_img_list = []
        #     for i in range(len(self.audiotype_map)):
        #         label = [k for k, v in self.audiotype_map.items() if v == i][0]
        #         event_image_path = os.path.join(self.event_image_path, label)
        #         aa = os.listdir(event_image_path)
        #         event_image_paths = [ 
        #             os.path.join(event_image_path,f) for f in aa
        #             if os.path.isfile(os.path.join(event_image_path,f)) and os.path.splitext(f)[1] in ['.npy'] 
        #         ]
        #         self.event_img_list.append(event_image_paths)
        # else:
        #     self.event_image_path = None

    def __len__(self):
        return len(self.text)
    
    def character_padding_forinput(self, img, img_length):
        """
        Args:
            img (np.array): image data
            img_length (list): list of character length
        Returns:
            np.array: padded image data
        """
        w = 0
        img_connected = img[:,w:w+img_length[0]]
        pleft = int(int((self.width-img_length[0])/2) + (self.width-img_length[0])%2)
        pright = int((self.width-img_length[0])/2)
        img_connected = np.pad(img_connected, [(0,0),(pleft,pright)], mode='constant', constant_values=255)
        w = w+img_length[0]
        for character_len in img_length[1:]:
            img_extract = img[:, w:w+character_len]
            pleft = int(int((self.width-character_len)/2) + (self.width-character_len)%2)
            pright = int((self.width-character_len)/2)
            img_extract = np.pad(img_extract, [(0,0),(pleft,pright)], mode='constant', constant_values=255)
            img_connected = cv2.hconcat([img_connected,img_extract])
            w=w+character_len
        return img_connected
    
    def __getitem__(self, idx):
        #basic info
        basename = self.basename[idx]           
        audiotype = self.audiotype[idx]
        audiotype_id = self.audiotype_map[audiotype]
        tmp_text = self.text[idx].replace("{", "").replace("}", "").replace("\n","")
        text = np.array([self.symbol_to_id[t] for t in list(tmp_text)])
        #load features
        mel = np.load(self.preprocessed_path/"mel"/audiotype/f"{basename}.npy")
        energy = np.load(self.preprocessed_path/"energy"/audiotype/f"{basename}.npy") if self.is_energy else None
        kurtosis = np.load(self.preprocessed_path/"kurtosis"/audiotype/f"{basename}.npy") if self.is_kurtosis else None
        duration = np.load(self.preprocessed_path/"duration"/audiotype/f"{basename}.npy")
        if self.use_image:
            image_length = np.load(self.preprocessed_path/"image"/"width"/audiotype/f"{basename}.npy").astype(np.int32)
            if np.max(image_length) > self.width:
                print(f"image length is over {self.width} pixels. {basename}")
            scale = cv2.IMREAD_GRAYSCALE if self.image_loadscale == "gray-scale" else cv2.IMREAD_COLOR
            image = cv2.imread(str(self.preprocessed_path/"image"/"png"/audiotype/f"{basename}.png"), scale)      
            image = self.character_padding_forinput(image, image_length)
            sample = {
                "id": basename,
                "audiotype": audiotype_id,
                "text": text,
                "mel": mel,
                "energy": energy,
                "kurtosis": kurtosis,
                "duration": duration,
                "image":image,
                "event_image_feature" : None
            }
        else:
            sample = {
            "id": basename,
            "audiotype": audiotype_id,
            "text": text,
            "mel": mel,
            "energy": energy,
            "duration": duration,
            "image":None,
            "event_image_feature":None
            }
        return sample

    def process_meta(self, filename):
        with open(self.preprocessed_path/filename, "r", encoding="utf-8") as f:
            filename = []
            audiotype = []
            fonttype = []
            fontsize = []
            text = []
            for line in f.readlines():
                fn, at, fs, ft, r = line.strip("\n").split("|")
                # あははもじもじ_24pt_c1-bells3-037-0980-30|bells3|24|あははもじもじ|チャリリリッリ
                filename.append(fn)
                audiotype.append(at)
                fonttype.append(ft)
                fontsize.append(fs)
                text.append(r)
            return filename, audiotype, fonttype, fontsize, text

    def reprocess(self, data, idxs):
        ids = [data[idx]["id"] for idx in idxs]                 
        audiotypes = [data[idx]["audiotype"] for idx in idxs]       
        texts = [data[idx]["text"] for idx in idxs]
        mels = [data[idx]["mel"] for idx in idxs]
        energies = [data[idx]["energy"] for idx in idxs]
        kurtosises = [data[idx]["kurtosis"] for idx in idxs]
        durations = [data[idx]["duration"] for idx in idxs]
        text_lens = np.array([text.shape[0] for text in texts])
        mel_lens = np.array([mel.shape[0] for mel in mels])
        audiotypes = np.array(audiotypes)
        texts = pad_1D(texts)
        mels = pad_2D(mels)
        energies = pad_1D(energies) if energies[0] is not None else None
        kurtosises = pad_1D(kurtosises) if kurtosises[0] is not None else None
        durations = pad_1D(durations)
        images=[data[idx]["image"] for idx in idxs]
        if images is not None and self.use_image:
            if self.image_loadscale == "gray-scale":
                images = pad_2D_gray_image(images, self.width, self.stride)
            elif self.image_loadscale == "RGB-scale":
                images=pad_2D_image(images, self.width, self.stride)
        else:
            images=None  
        event_image_features=np.array([data[idx]["event_image_feature"] for idx in idxs])
        response=[
            ids,audiotypes,texts,text_lens,max(text_lens),
            mels,mel_lens,max(mel_lens),
            energies,kurtosises,durations,
            images,event_image_features
        ]
        return tuple(response)

    def collate_fn(self, data):
        data_size = len(data)
        if self.sort:
            len_arr = np.array([d["text"].shape[0] for d in data])
            idx_arr = np.argsort(-len_arr)
        else:
            idx_arr = np.arange(data_size)
        tail = idx_arr[len(idx_arr) - (len(idx_arr) % self.batch_size) :]
        idx_arr = idx_arr[: len(idx_arr) - (len(idx_arr) % self.batch_size)]
        idx_arr = idx_arr.reshape((-1, self.batch_size)).tolist()
        if not self.drop_last and len(tail) > 0:
            idx_arr += [tail.tolist()]
        output = list()
        for idx in idx_arr:
            output.append(self.reprocess(data, idx))
        return output
    
def pil2cv(pil_im, color=False):
    ''' PIL型 -> OpenCV型 '''
    new_image = np.array(pil_im, dtype=np.uint8)
    if new_image.ndim == 2:  # モノクロ
        pass
    elif new_image.shape[2] == 3:  # カラー
        if color:
            new_image = cv2.cvtColor(new_image, cv2.COLOR_RGB2BGR)
        else:
            new_image = cv2.cvtColor(new_image, cv2.COLOR_RGB2GRAY)
    elif new_image.shape[2] == 4:  # 透過
        if color:
            new_image = cv2.cvtColor(new_image, cv2.COLOR_RGBA2BGRA)
        else:
            new_image = cv2.cvtColor(new_image, cv2.COLOR_RGBA2GRAY)
    return new_image

def _draw_image(text, font_path, font_size, max_width, font_width, bgcolor, textcolor):
    # font
    font_width = int(font_width)
    _image = PIL.Image.new("RGB", [100, 100], bgcolor)
    _draw = PIL.ImageDraw.Draw(_image)
    font = PIL.ImageFont.truetype(os.path.join(font_path,"あははもじもじ","あははもじもじ.otf"), font_size)
    text_width, text_height = _draw.textsize(text, font=font)
    image_assessment = PIL.Image.new("RGB", [font_width*len(text), font_size], bgcolor)
    image_padded = PIL.Image.new("RGB", [max_width*len(text), font_size], bgcolor)

    # config padding
    def img_pad(im, max_width):
        def add_margin(pil_img, top, right, bottom, left, color):
            width, height = pil_img.size
            new_width = width + right + left
            new_height = height + top + bottom
            result = PIL.Image.new(pil_img.mode, (new_width, new_height), color)
            result.paste(pil_img, (left, top))
            return result
        pad_left = (max_width - im.width)/2 + ((max_width - im.width)%2)
        pad_right = (max_width - im.width)/2

        return add_margin(im, 0, int(pad_right), 0, int(pad_left), (0,0,0))

    # create visual-text
    w = 0
    for i,char in enumerate(text):
        # canvas
        image_assessment_tmp = PIL.Image.new("RGB", [font_size, font_size], bgcolor)
        draw = PIL.ImageDraw.Draw(image_assessment_tmp)
        # draw text
        draw.text((0, 0), char, fill=textcolor, font=font)
        # expansion or contraction
        image_assessment_tmp = image_assessment_tmp.resize((font_width, font_size))

        image_assessment.paste(image_assessment_tmp, (w,0))

        # image for model input
        image_padded_tmp = img_pad(image_assessment_tmp.copy(), max_width)
        image_padded.paste(image_padded_tmp, (max_width*i, 0))
        w += font_width
    # save
    assert image_assessment.size[0]%font_width==0, f"image_assessment is {image_assessment.size}"
    assert image_assessment.size[1]%font_size==0, f"image_assessment is {image_assessment.size}"
    return pil2cv(image_assessment), pil2cv(image_padded)

class TestDataset(Dataset):
    def __init__(self, filepath, preprocess_config, train_config, sort=True, gt=True, fontsize=None):
        #path
        self.preprocessed_path = preprocess_config["path"]["preprocessed_data_path"]
        self.input_type = preprocess_config["preprocessing"]["input_type"]
        self.symbol_to_id = get_symbols(self.preprocessed_path)
        self.sort = sort
        self.gt = gt
        self.base_fontsize = fontsize

        #get basic info of test data
        self.basename, self.audiotype, self.fontsize, self.fonttype, self.text = self.process_meta(
            filepath
        )
        self.font_dir = preprocess_config["path"]["font_path"]
        #image information
        self.use_image = train_config["use_image"]
        self.use_image_encoder = train_config["image_encoder"]
        if self.input_type == 'visual-text':
            #image information
            self.text_font_sizes = preprocess_config["preprocessing"]["text"]["font_size"]
            self.text_font_name = preprocess_config["preprocessing"]["text"]["font_size"]
            self.image_bgcolor = preprocess_config["preprocessing"]["image"]["background_color"]
            self.image_textcolor = preprocess_config["preprocessing"]["image"]["text_color"]
            self.image_loadscale = preprocess_config["preprocessing"]["image"]["load_scale"]
            with open(os.path.join(preprocess_config["path"]["preprocessed_data_path"], "visual_text.json")) as f:
                visual_text_info = json.load(f)
            self.width = visual_text_info["max_pixelsize"][0]
            self.height = visual_text_info["height"][0]
            self.stride = visual_text_info["max_pixelsize"][0]

        #audiotype id function
        with open(os.path.join(self.preprocessed_path, "audiotype.json")) as f:
            self.audiotype_map = json.load(f)
        
        self.use_image_encoder = train_config["image_encoder"]
        if self.use_image_encoder:
            self.event_image_path = preprocess_config["path"]["event_image_path"]
            self.event_img_list = []
            for i in range(len(self.audiotype_map)):
                label = [k for k, v in self.audiotype_map.items() if v == i][0]
                event_image_path = os.path.join(self.event_image_path, label)
                aa = os.listdir(event_image_path)
                event_image_paths = [ 
                    os.path.join(event_image_path,f) for f in aa
                    if os.path.isfile(os.path.join(event_image_path,f)) and os.path.splitext(f)[1] in ['.npy'] 
                ]
                self.event_img_list.append(event_image_paths)
        else:
            self.event_image_path = None

        #test batch
        self.data_num=len(self.basename)
        self.symbol_to_id = get_symbols(self.preprocessed_path)
        self.batchs=self.get_batch()

        

    def get_batch(self):
        batchs=[]
        for idx in tqdm(range(self.data_num)):
            #id
            tmp = self.basename[idx].split('_')
            if self.base_fontsize is not None:
                base_ = '_'.join([tmp[2],str(self.base_fontsize),self.text[idx]])
                ids = [base_]
            else:
                base_ = '_'.join([tmp[2],str(self.fontsize[idx]),self.text[idx]])
                ids = [base_]

            #audiotype
            audiotype = self.audiotype[idx]
            audiotype_id = np.array([self.audiotype_map[audiotype]])
            
            #texts
            texts = np.array([[self.symbol_to_id[t] for t in list(self.text[idx].replace("{", "").replace("}", ""))]])

            #text lens
            text_lens = np.array([len(texts[0])])

            if self.gt:
                #load mel
                mel_path = os.path.join(
                    self.preprocessed_path,
                    "mel",
                    audiotype,
                    "{}.npy".format(self.basename[idx]), 
                )
                mel = np.array([np.load(mel_path)])
                mel_len = np.array([mel[0].shape[0]])

                # load energy
                energy_path = os.path.join(
                    self.preprocessed_path,
                    "energy",
                    audiotype,
                    "{}.npy".format(self.basename[idx]),
                )
                energy = np.array([np.load(energy_path)])

                # load duration
                duration_path = os.path.join(
                    self.preprocessed_path,
                    "duration",
                    audiotype,
                    "{}.npy".format(self.basename[idx]),
                )
                duration = np.load(duration_path)
                duration = np.array([duration.astype(np.int32)])

                #image
                if self.use_image:
                    image_length_path = os.path.join(
                        self.preprocessed_path,
                        "image",
                        "width",
                        audiotype,
                        "{}.npy".format(self.basename[idx]),
                    )
                    image_length = np.load(image_length_path)
                    image_path= os.path.join(
                        self.preprocessed_path,
                        "image",
                        "png",
                        audiotype,
                        "{}.png".format(self.basename[idx]),
                    )
                    image=cv2.imread(image_path,cv2.IMREAD_GRAYSCALE)
                    image = self.character_padding_forinput(image, image_length)
                    image = [image]
                    if self.use_image_encoder:
                        assert False, "not implement. resolve random process."
                else:
                    image=None

                #              ids, audiotype_id, texts, text_lens, max(text_lens), mel, mel_len, max(mel_len), energy, duration, image))
                batchs.append((ids, audiotype_id, texts, text_lens, max(text_lens), mel, mel_len, max(mel_len), energy, duration, image, [None]))
            else:
                #load mel
                mel = None
                mel_len = None
                # load energy
                energy = None
                # load duration
                duration = None
                #image
                if self.use_image:
                    if self.base_fontsize is not None:
                        fs = self.base_fontsize
                    else:
                        fs = self.fontsize[idx]
                    image_assessment, image = _draw_image(
                        text=self.text[idx],
                        font_path=self.font_dir,
                        font_size=24,
                        max_width=self.width,
                        font_width=fs,
                        bgcolor=tuple(self.image_bgcolor),
                        textcolor=tuple(self.image_textcolor)
                    )
                    image = [image]
                else:
                    image = None
                batchs.append((ids, audiotype_id, texts, text_lens, max(text_lens), mel, mel_len, None, energy, duration, image, [None]))
        return batchs

    def process_meta(self, filename):
        with open(
            os.path.join(self.preprocessed_path, filename), "r", encoding="utf-8"
        ) as f:
            filename = []
            audiotype = []
            fonttype = []
            fontsize = []
            text = []
            for line in f.readlines():
                fn, at, fs, ft, r = line.strip("\n").split("|")
                filename.append(fn)
                audiotype.append(at)
                fontsize.append(fs)
                fonttype.append(ft)
                text.append(r)
            return filename, audiotype, fontsize, fonttype, text

class TestDataset_img(Dataset):
    def __init__(self, filepath, preprocess_config, train_config, event_img_path=None, sort=True, gt=True):
        #path
        self.preprocessed_path = preprocess_config["path"]["preprocessed_data_path"]
        self.input_type = preprocess_config["preprocessing"]["input_type"]
        self.symbol_to_id = get_symbols(self.preprocessed_path)
        self.sort = sort
        self.gt = gt
        self.event_image_path = event_img_path


        #get basic info of test data
        self.basename, self.audiotype, self.fontsize, self.fonttype, self.text, self.img_enc = self.process_meta(
            filepath
        )
        self.font_dir = preprocess_config["path"]["font_path"]
        #image information
        self.use_image = train_config["use_image"]
        self.use_image_encoder = train_config["image_encoder"]
        if self.input_type == 'visual-text':
            #image information
            self.text_font_sizes = preprocess_config["preprocessing"]["text"]["font_size"]
            self.text_font_name = preprocess_config["preprocessing"]["text"]["font_size"]
            self.image_bgcolor = preprocess_config["preprocessing"]["image"]["background_color"]
            self.image_textcolor = preprocess_config["preprocessing"]["image"]["text_color"]
            self.image_loadscale = preprocess_config["preprocessing"]["image"]["load_scale"]
            with open(os.path.join(preprocess_config["path"]["preprocessed_data_path"], "visual_text.json")) as f:
                visual_text_info = json.load(f)
            self.width = visual_text_info["max_pixelsize"][0]
            self.height = visual_text_info["height"][0]
            self.stride = visual_text_info["max_pixelsize"][0]

        #audiotype id function
        with open(os.path.join(self.preprocessed_path, "audiotype.json")) as f:
            self.audiotype_map = json.load(f)
        
        self.use_image_encoder = train_config["image_encoder"]

        #test batch
        self.data_num=len(self.basename)
        self.symbol_to_id = get_symbols(self.preprocessed_path)
        self.batchs=self.get_batch()

        

    def get_batch(self):
        batchs=[]
        for idx in tqdm(range(self.data_num)):
            #id
            tmp = self.basename[idx].split('_')
            base_ = '_'.join([tmp[2],tmp[1],self.text[idx]])
            ids = [base_]

            #audiotype
            audiotype = self.audiotype[idx]
            audiotype_id = np.array([self.audiotype_map[audiotype]])
            
            #texts
            texts = np.array([[self.symbol_to_id[t] for t in list(self.text[idx].replace("{", "").replace("}", ""))]])

            #text lens
            text_lens = np.array([len(texts[0])])

            if self.gt:
                #load mel
                mel_path = os.path.join(
                    self.preprocessed_path,
                    "mel",
                    audiotype,
                    "{}.npy".format(self.basename[idx]), 
                )
                mel = np.array([np.load(mel_path)])
                mel_len = np.array([mel[0].shape[0]])

                # load energy
                energy_path = os.path.join(
                    self.preprocessed_path,
                    "energy",
                    audiotype,
                    "{}.npy".format(self.basename[idx]),
                )
                energy = np.array([np.load(energy_path)])

                # load duration
                duration_path = os.path.join(
                    self.preprocessed_path,
                    "duration",
                    audiotype,
                    "{}.npy".format(self.basename[idx]),
                )
                duration = np.load(duration_path)
                duration = np.array([duration.astype(np.int32)])

                #image
                if self.use_image:
                    image_length_path = os.path.join(
                        self.preprocessed_path,
                        "image_length",
                        audiotype,
                        "{}.npy".format(self.basename[idx]),
                    )
                    image_length = np.load(image_length_path)
                    image_path= os.path.join(
                        self.preprocessed_path,
                        "image",
                        audiotype,
                        "{}.png".format(self.basename[idx]),
                    )
                    image=cv2.imread(image_path,cv2.IMREAD_GRAYSCALE)
                    image = self.character_padding_forinput(image, image_length)
                    image = [image]
                    if self.use_image_encoder:
                        event_img_feature_path = Path(self.event_image_path) / audiotype / self.img_enc[idx]
                        event_img_feature = np.array([np.load(event_img_feature_path)])
                        aa = os.path.splitext(os.path.basename(event_img_feature_path))[0]
                        aa = aa.split("_")[-1]
                        aa = aa.strip("_")
                        tmp = ids[0]
                        tmp = tmp.split("_")
                        ids[0] = f'{tmp[0]}_{tmp[1]}_event{aa}_{tmp[2]}'
                    else:
                        event_img_feature = [None]
                else:
                    image=None

                #              ids, audiotype_id, texts, text_lens, max(text_lens), mel, mel_len, max(mel_len), energy, duration, image))
                batchs.append((ids, audiotype_id, texts, text_lens, max(text_lens), mel, mel_len, max(mel_len), energy, duration, image, event_img_feature))
            else:
                #load mel
                mel = None
                mel_len = None
                # load energy
                energy = None
                # load duration
                duration = None
                #image
                if self.use_image:
                    image_assessment, image = _draw_image(
                        text=self.text[idx],
                        font_path=self.font_dir,
                        font_size=24,
                        max_width=self.width,
                        font_width=self.fontsize[idx],
                        bgcolor=tuple(self.image_bgcolor),
                        textcolor=tuple(self.image_textcolor)
                    )
                    image = [image]
                else:
                    image = None
                batchs.append((ids, audiotype_id, texts, text_lens, max(text_lens), mel, mel_len, None, energy, duration, image, [None]))
        return batchs

    def process_meta(self, filename):
        with open(
            os.path.join(self.preprocessed_path, filename), "r", encoding="utf-8"
        ) as f:
            filename = []
            audiotype = []
            fonttype = []
            fontsize = []
            text = []
            img_enc = []
            for line in f.readlines():
                fn, at, fs, ft, r, en = line.strip("\n").split("|")
                filename.append(fn)
                audiotype.append(at)
                fontsize.append(fs)
                fonttype.append(ft)
                text.append(r)
                img_enc.append(en)
            return filename, audiotype, fontsize, fonttype, text, img_enc
