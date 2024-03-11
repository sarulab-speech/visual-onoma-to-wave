from pathlib import Path
from tqdm import tqdm
import librosa
import numpy as np
import os
import json
import tgt
import torchaudio
import torch
from joblib import Parallel, delayed
from .visualtext_generator import Generator
from PIL import Image
from sklearn.preprocessing import StandardScaler
import random
import shutil

class Preprocessor:
    def __init__(self,config):
        self.config = config
        self._config_open(config)
        self.path_preprocessed.mkdir(parents=True, exist_ok=True)
        self.spec_module = torchaudio.transforms.Spectrogram(
            n_fft=self.filter_length,
            win_length=self.win_length,
            hop_length=self.hop_length,
            power=1,
            center=True,
        )
        self.mel_scale = torchaudio.transforms.MelScale(
            n_mels=self.n_mel_channels,
            sample_rate=self.sampling_rate,
            f_min=self.mel_fmin,
            f_max=self.mel_fmax,
            n_stft=self.filter_length // 2 + 1,
            norm="slaney"
        )
        with open(os.path.join(self.path_formatted, "dataset_length.json"), "r") as f:
            self.text_length_info = json.load(f)

    def build_from_path(self, num_workers=10):
        if len(self.extract_labels) > 0: # use specified labels
            self.labels = sorted(list(
                set([x.parent.stem for x in (self.path_formatted / "audio").glob("*/*.wav")]) &
                set(self.extract_labels)
            ))
        else: # use all labels
            self.labels = sorted(list(
                set([x.parent.stem for x in (self.path_formatted / "audio").glob("*/*.wav")])
            ))       
        audio_labels = {}
        width_dumps = {} # {label: character-len_per_1sec}
        info_list_list = []
        wav_lens_list = []
        n_frames_cnt = 0
        print("===Extracting features===")
        outer_bar = tqdm(total=len(self.labels), desc="Extracting", position=0)
        # compute mel-spectrogram, duration, energy
        for i, label in enumerate(self.labels):
            audio_labels[label] = i # label to id
            with open(self.path_formatted / "text"/ label / "data.txt", "r") as f:
                info_list = f.readlines()
            self._makedirs(label)
            # joblib
            results = Parallel(n_jobs=num_workers, verbose=1)(
                delayed(self._process)(label, line) for line in info_list
            )
            results = np.array(results)
            # del -1
            mel_lens, wav_lens, text_lens = zip(*results)
            mel_lens, wav_lens, text_lens = np.array(mel_lens), np.array(wav_lens), np.array(text_lens)
            del_idx = np.where(mel_lens==-1)[0]
            all_ = len(info_list)
            info_list, mel_lens, wav_lens, text_lens = np.delete(info_list, del_idx), np.delete(mel_lens, del_idx), np.delete(wav_lens, del_idx), np.delete(text_lens, del_idx)
            computed_info = self._compute_visualtextinfo(wav_lens, text_lens)
            width_dumps[label] = computed_info
            info_list_list.append(info_list)
            wav_lens_list.append(wav_lens)
            print(f"label: {label}, del_num/all: {len(del_idx)}/{all_}")
            n_frames_cnt += mel_lens.sum()
            outer_bar.update(1)       
        with open(os.path.join(self.path_preprocessed, "audiotype.json"), "w") as f:
            f.write(json.dumps(audio_labels))
        with open(os.path.join(self.path_preprocessed, "label_width.json"), "w") as f:
            f.write(json.dumps(width_dumps))
        # generate visual onomatopoeia
        print("===Generating visual onomatopoeia===")
        entire_max_width = 0
        for i, info_list, wav_lens in zip(range(len(info_list_list)), info_list_list, wav_lens_list):
            label = [k for k, v in audio_labels.items() if v == i][0]
            character_persec, max_width, min_width = width_dumps[label]
            Visual_ono_Generator = Generator(self.config, character_persec, max_width)
            Parallel(n_jobs=num_workers, verbose=1)(
                delayed(self._process_visual_ono)(label, info, wav_len, Visual_ono_Generator) for info, wav_len in zip(info_list, wav_lens)
            )
            entire_max_width = max(entire_max_width, max_width)
        with open(os.path.join(self.path_preprocessed, "visual_text.json"), "w") as f:
            stats = {
                "max_pixelsize": [
                    int(entire_max_width)
                ],
                "height": [
                    self.im_fontsize
                ]
            }
            f.write(json.dumps(stats))
        print("====Start augmentation====")
        for label, info_list in zip(self.labels, info_list_list):
            results = Parallel(n_jobs=num_workers, verbose=1)(
                delayed(self._augmentation)(label, info) for info in info_list
            )
            mel_lens = np.array(results)
            n_frames_cnt += mel_lens.sum()
        print("===feature normalization===")
        energy_scaler = StandardScaler()
        kurtosis_scaler = StandardScaler()
        energy_npy_list, kurtosis_npy_list = list(Path(self.path_preprocessed / "energy").glob("*/*.npy")), list(Path(self.path_preprocessed / "kurtosis").glob("*/*.npy")) 
        for energy_npy, kurtosis_npy in tqdm(zip(energy_npy_list, kurtosis_npy_list), total=len(energy_npy_list)):
            energy = np.load(energy_npy)
            kurtosis = np.load(kurtosis_npy)
            energy = self._remove_outlier(energy)
            kurtosis = self._remove_outlier(kurtosis)
            if len(energy) > 0:
                energy_scaler.partial_fit(energy.reshape(-1, 1))
            if len(kurtosis) > 0:
                kurtosis_scaler.partial_fit(kurtosis.reshape(-1, 1))
        energy_mean, energy_std = energy_scaler.mean_[0], energy_scaler.scale_[0]
        kurtosis_mean, kurtosis_std = kurtosis_scaler.mean_[0], kurtosis_scaler.scale_[0]
        energy_min, energy_max = self._normalize(self.path_preprocessed / "energy", energy_mean, energy_std)
        kurtosis_min, kurtosis_max = self._normalize(self.path_preprocessed / "kurtosis", kurtosis_mean, kurtosis_std)
        with open(self.path_preprocessed / "stats.json", "w") as f:
            f.write(json.dumps({
                "energy": [
                    float(energy_min),
                    float(energy_max),
                    float(energy_mean),
                    float(energy_std)
                ],
                "kurtosis": [
                    float(kurtosis_min),
                    float(kurtosis_max),
                    float(kurtosis_mean),
                    float(kurtosis_std)
                ]
            }))
        print("===write metadata===")
        train_info_list = list(Path(self.path_preprocessed / "intermediate" / "info" / "train").glob("*/*.txt"))
        with open(self.path_preprocessed / "train.txt", "w") as f:
            for info in train_info_list:
                f.write(info.read_text())
                f.write("\n")
        val_test_info_list = list(Path(self.path_preprocessed / "intermediate" / "info" / "val_test").glob("*/*.txt"))
        random.shuffle(val_test_info_list)
        val_list, test_list = val_test_info_list[:len(val_test_info_list)//2], val_test_info_list[len(val_test_info_list)//2:]
        with open(self.path_preprocessed / "val.txt", "w") as f:
            for info in val_list:
                f.write(info.read_text())
                f.write("\n")
        with open(self.path_preprocessed / "test.txt", "w") as f:
            for info in test_list:
                f.write(info.read_text())
                f.write("\n")
        time_ = n_frames_cnt*self.hop_length/self.sampling_rate / 3600
        print(f"===feature extraction finished. {time_} hours===")
        # rm intermediate files
        shutil.rmtree(self.path_preprocessed / "intermediate")

    def _config_open(self, config_dict):
        """ Open config file. 
        Args:
            config_dict (dict): config file dictionary
        """
        # path
        self.path_corpus, self.path_formatted = Path(config_dict["path"]["corpus"]), Path(config_dict["path"]["formatted"])
        self.path_preprocessed, self.path_font = Path(config_dict["path"]["preprocessed"]), Path(config_dict["path"]["font"])

        # dataset
        self.extract_labels, self.valtest_id = config_dict["dataset"]["extract_labels"], config_dict["dataset"]["valtest_id"]
        self.confidence_score_border, self.acceptance_score_border = config_dict["dataset"]["confidence_score_border"], config_dict["dataset"]["acceptance_score_border"]

        self.input_type = config_dict["input_type"]

        # visual text
        self.im_fontsize = config_dict["visual_text"]["fontsize"]
        self.im_is_stretching = config_dict["visual_text"]["image_stretching"]
        self.im_bgcolor = config_dict["visual_text"]["color"]["background"]
        self.im_txtcolor = config_dict["visual_text"]["color"]["text"]
        self.im_padcolor = config_dict["visual_text"]["color"]["pad"]
        self.im_loadscale = config_dict["visual_text"]["scale_in_training"]

        # audio
        self.sampling_rate = config_dict["audio"]["sampling_rate"]
        self.max_wav_value = config_dict["audio"]["max_wav_value"]
        self.filter_length = config_dict["audio"]["stft"]["filter_length"]
        self.hop_length = config_dict["audio"]["stft"]["hop_length"]
        self.win_length = config_dict["audio"]["stft"]["win_length"]
        self.margin_frame = config_dict["audio"]["stft"]["margin_frame"]
        ## mel
        self.n_mel_channels = config_dict["audio"]["mel"]["n_mel_channels"]
        self.mel_fmin = config_dict["audio"]["mel"]["mel_fmin"]
        self.mel_fmax = config_dict["audio"]["mel"]["mel_fmax"]
        ## feature
        self.energy_is_norm = config_dict["audio"]["feature"]["energy"]["normalization"]
        self.kurt_is_norm = config_dict["audio"]["feature"]["kurtosis"]["normalization"]

        # augmentation
        self.aug_maxlen = config_dict["augmentation"]["max_length"]
        self.aug_repeatnum = config_dict["augmentation"]["repeat_num"]
        self.aug_chara_consecutive_num = config_dict["augmentation"]["consecutive_num"]
        self.aug_first_consecutive = config_dict["augmentation"]["first_consecutive"]

    def _makedirs(self, label):
        """ Make directories for preprocessed data.
        Args:
            label (str): label name
        """
        _path = self.path_preprocessed / "duration" / label
        _path.mkdir(parents=True, exist_ok=True)
        _path = self.path_preprocessed / "energy" / label
        _path.mkdir(parents=True, exist_ok=True)
        _path = self.path_preprocessed / "kurtosis" / label
        _path.mkdir(parents=True, exist_ok=True)
        _path = self.path_preprocessed / "mel" / label
        _path.mkdir(parents=True, exist_ok=True)
        _path = self.path_preprocessed / "image" / "png" / label
        _path.mkdir(parents=True, exist_ok=True)
        _path = self.path_preprocessed / "image" / "width" / label
        _path.mkdir(parents=True, exist_ok=True)
        _path = self.path_preprocessed / "intermediate" / "info" / "train" / label
        _path.mkdir(parents=True, exist_ok=True)
        _path = self.path_preprocessed / "intermediate" / "info" / "val_test" / label
        _path.mkdir(parents=True, exist_ok=True)

    def _check_score_border(self, confidence_score, acceptance_score):
        """ Check score border.
        Args:
            confidence_score (float): confidence score
            acceptance_score (float): acceptance score
        Returns:
            bool: True if both scores are over the border
        """
        if float(confidence_score) < self.confidence_score_border:
            return False
        if float(acceptance_score) < self.acceptance_score_border:
            return False
        return True

    def _get_basename(self, font_name, font_size, stem, ext=".png"):
        """ Get basename. -> {font_name}_{font_size}pt_{stem}{ext}
        Underline replaced by hyphen because basename is splited by underline.
        Args:
            font_name (str): font name
            font_size (int): font size
            stem (str): stem
            ext (str): extension
        Returns:
            str: basename
        """
        base = stem.replace(" ", "").replace("_", "-")
        return f"{font_name}_{font_size}pt_{base}{ext}"

    def _get_alignment(self, tier, wav, margin_frame):
        """ Get alignment.
        Args:
            tier (TextGridTier): tier
            wav (np.ndarray): wav
            margin_frame (int): margin frame
        Returns:
            list: phones
            list: durations
            float: start time
            float: end time
        """
        sil_phones = ["sil", "sp", "spn", 'silB', 'silE', '']
        phones, durations = [], []
        start_t, end_t, end_idx = 0, 0, 0
        wav_sec, margin_sec = len(wav)/self.sampling_rate, margin_frame*self.hop_length/self.sampling_rate
        starts_np, ends_np = np.array([]), np.array([])
        # open tier
        for t in tier._objects:
            s, e, p = t.start_time, t.end_time, t.text
            # first
            if len(phones) == 0:
                if p in sil_phones:
                    continue
                else:
                    start_t = s
            # ordinary phones
            if p not in sil_phones:
                phones.append(p)
                end_t = e
                end_idx = len(phones)
            else:
                phones.append('sp')
                last_t = e
            starts_np = np.append(starts_np, s)
            ends_np = np.append(ends_np, e)
        start_t, end_t, last_t = start_t*wav_sec/last_t, end_t*wav_sec/last_t, last_t*wav_sec/last_t
        starts_np, ends_np = starts_np*wav_sec/last_t, ends_np*wav_sec/last_t
        # margin
        if start_t-margin_sec < 0:
            start_t = 0
        else:
            start_t = start_t-margin_sec
        starts_np[0] = start_t
        if end_t+margin_sec > last_t:
            end_t = last_t
        else:
            end_t = end_t+margin_sec
        ends_np[-2] = end_t
        for i in range(starts_np.shape[0]):
            s = starts_np[i]
            e = ends_np[i]
            durations.append(
                int(
                    np.round(e * self.sampling_rate /
                             self.hop_length)
                    - np.round(s * self.sampling_rate /
                               self.hop_length)
                )
            )
        phones, durations = phones[:end_idx], durations[:end_idx]
        return phones, durations, start_t, end_t

    def _get_spec(self, audio):
        """ Get spectrogram.
        Args:
            audio (np.ndarray): audio
        Returns:
            np.ndarray: mel spectrogram
            np.ndarray: energy
        """
        audio = torch.clip(torch.from_numpy(audio), -1, 1)
        magspec = self.spec_module(audio)
        melspec = self.mel_scale(magspec)
        logmelspec = torch.log(torch.clamp_min(
            melspec, 1.0e-5) * 1.0).to(torch.float32)
        energy = torch.norm(magspec, dim=0)
        return logmelspec.numpy(), energy.numpy()

    def _get_kurtosis(self, audio: np.ndarray, duration: list):
        """ Get kurtosis.
        Args:
            audio (np.ndarray): audio
            duration (list): duration
        Returns:
            np.ndarray: kurtosis
        """
        eps = 1e-8
        audio = torch.clip(torch.from_numpy(audio), -1, 1)
        duration = [0] + list(duration)
        powerspec = self.spec_module(audio)**2
        kurtosises = np.zeros(len(duration)-1)
        for i in range(len(duration)-1):
            spec_tmp = powerspec[:, sum(duration[:i+1]): sum(duration[:i+2])]
            gamma = torch.log(torch.mean(spec_tmp)+eps) - torch.mean(torch.log(spec_tmp+eps))
            eta = (3-gamma+torch.sqrt((gamma-3)**2+24*gamma))/(12*gamma)
            kurtosises[i] = (eta+2)*(eta+3)/(eta*(eta+1)+eps)
        return kurtosises

    def _is_traindata(self, savename):
        audio_numbering = int(savename.split('-')[2])
        if audio_numbering not in self.valtest_id:
            return True
        else:
            return False

    def _process(self, label, line):
        """ Process.
        Args:
            label (str): audio label
            line (str): {onomatopoeia_name}|{audio_name}|{onomatopoeia}|{audio_label}|{confidence_score}|{acceptance_score}
        Returns:
            int: mel spectrogram length
            int: wav length
            int: text length
        """
        text_base, audio_base, text, _, confidence_score, acceptance_score = line.replace('\n','').split("|")
        # check score border
        if not self._check_score_border(confidence_score, acceptance_score):
            return -1,-1,-1
        tg_path = self.path_formatted/"TextGrid"/label/f"{text_base}.TextGrid"
        basename = self._get_basename(self.path_font.stem, self.im_fontsize, text_base, ext="")
        if not tg_path.exists():
            return -1,-1,-1
        wav_path = self.path_formatted/"audio"/label/f"{audio_base}.wav"
        wav, _ = librosa.load(str(wav_path), sr=22050)
        textgrid = tgt.io.read_textgrid(tg_path)
        character, duration, start, end = self._get_alignment(
            textgrid.get_tier_by_name("phones"),wav,self.margin_frame)
        if len(character) != len(text):
            return -1,-1,-1
        trimed_wav = wav[int(self.sampling_rate * start):].astype(np.float32)
        wav_len = len(trimed_wav)
        if start >= end:
            return -1,-1,-1
        if len(wav[int(self.sampling_rate * start): int(self.sampling_rate * end)]) < len(wav)/15:
            return -1,-1,-1
        mel_spectrogram, energy = self._get_spec(trimed_wav)
        mel_spectrogram = mel_spectrogram[:, : sum(duration)]
        energy = energy[: sum(duration)]
        # energy: frame-level to character-level
        pos = 0
        for i, d in enumerate(duration):
            if d > 0:
                energy[i] = np.mean(energy[pos: pos + d])
            else:
                energy[i] = 0
            pos += d
        energy = energy[: len(duration)]
        # get kurtosis
        kurtosis = self._get_kurtosis(trimed_wav, duration)
        np.save(self.path_preprocessed/"kurtosis"/label/f"{basename}.npy", kurtosis)
        np.save(self.path_preprocessed/"duration"/label/f"{basename}.npy", duration)
        np.save(self.path_preprocessed/"energy"/label/f"{basename}.npy", energy)
        np.save(self.path_preprocessed/"mel"/label/f"{basename}.npy", mel_spectrogram.T)
        info = f"{basename}|{label}|{self.im_fontsize}|{self.path_font.stem}|{text}"
        if self._is_traindata(basename):
            with open(os.path.join(self.path_preprocessed, "intermediate", "info", "train", label, f"{basename}.txt"), "w") as f:
                f.write(info)
        else:
            with open(os.path.join(self.path_preprocessed, "intermediate", "info", "val_test", label, f"{basename}.txt"), "w") as f:
                f.write(info)
        return mel_spectrogram.shape[1], wav_len, len(text)

    def _process_visual_ono(self, label, info, wav_len, Visual_ono_Generator):
        """ Process visual onomatopoeia.
        Args:
            label (str): audio label
            info (str): {basename}|{label}|{fontsize}|{fontname}|{text}
            wav_len (int): wav length
            Visual_ono_Generator (Generator): visual onomatopoeia generator
        """
        text_base, _, text, _, _, _ = info.replace('\n','').split("|")
        wav_sec = wav_len/self.sampling_rate
        # generate visual onomatopoeia
        save_im_path = os.path.join(self.path_preprocessed, "image", "png", label, self._get_basename(self.path_font.stem, self.im_fontsize, text_base, ext=".png"))
        save_width_path = os.path.join(self.path_preprocessed, "image", "width", label, self._get_basename(self.path_font.stem, self.im_fontsize, text_base, ext=".npy"))
        Visual_ono_Generator.draw(text, wav_sec, save_im_path, save_width_path)

    def _compute_visualtextinfo(self, wav_len, text_len):
        wav_sec = wav_len/self.sampling_rate
        character_len_per1sec_mean = np.mean(text_len/wav_sec)
        canvas_width = np.ceil(character_len_per1sec_mean*wav_sec*self.im_fontsize).astype(np.int32)
        character_width_max = np.max(np.ceil(canvas_width/text_len))
        character_width_min = np.min(np.ceil(canvas_width/text_len))
        return character_len_per1sec_mean, character_width_max, character_width_min

    def _exist_feature_load(self, label, basename):
        """
        Args:
            label (str): label
            basename (str): basename
        Returns:
            duration (np.ndarray): duration
            energy (np.ndarray): energy
            kurtosis (np.ndarray): kurtosis
            mel (np.ndarray): mel spectrogram
            image (PIL.Image): image
            width (np.ndarray): width
        """
        duration = np.load(self.path_preprocessed/"duration"/label/f"{basename}.npy")
        energy = np.load(self.path_preprocessed/"energy"/label/f"{basename}.npy")
        kurtosis = np.load(self.path_preprocessed/"kurtosis"/label/f"{basename}.npy")
        mel = np.load(self.path_preprocessed/"mel"/label/f"{basename}.npy").T
        image = Image.open(self.path_preprocessed/"image"/"png"/label/f"{basename}.png")
        width = np.load(self.path_preprocessed/"image"/"width"/label/f"{basename}.npy")
        return duration, energy, kurtosis, mel, image, width

    def _augmentation(self, label, info):
        text_base, _, text, _, _, _ = info.replace('\n','').split("|")
        basename = self._get_basename(self.path_font.stem, self.im_fontsize, text_base, ext="")
        aug_n_sum = 0
        # augmentation of repeated text
        repeat_num = 2
        while (repeat_num<=self.aug_repeatnum) and (len(text)<=self.aug_maxlen):
            savename = f"{basename}-repeat{repeat_num}"
            n = self._repeataug(repeat_num, label, basename, savename, text)
            repeat_num += 1
            aug_n_sum += n
        # augmentation of repeating first character (not included ICASSP2023 paper)
        augment_first_num = 1
        while (augment_first_num<=self.aug_first_consecutive) and (len(text)<=self.aug_maxlen):
            savename = f"{basename}-firstconsecutive{augment_first_num}"
            n = self._consecutiveaug(augment_first_num+1, 0, label, basename, savename, text)
            augment_first_num += 1
            aug_n_sum += n
        # augmentation of consecutive characters
        base_pos = self._get_consecutive_pos(text)
        augment_consecutive_num = 1
        while (augment_consecutive_num<=self.aug_chara_consecutive_num) and (len(text)<=self.aug_maxlen) and (base_pos is not None):
            savename = f"{basename}-consecutive{augment_consecutive_num}"
            n, ret_text = self._consecutiveaug(augment_consecutive_num+1, base_pos, label, basename, savename, text)
            augment_consecutive_num += 1
            aug_n_sum += n
            repeat_num = 2
            ret_basename = savename
            while (repeat_num<=self.aug_repeatnum) and (len(ret_text)<=self.aug_maxlen):
                savename = f"{ret_basename}-repeat{repeat_num}"
                n = self._repeataug(repeat_num, label, ret_basename, savename, ret_text)
                repeat_num += 1
                aug_n_sum += n
        return aug_n_sum

    def _repeataug(self, repeat_num, label, basename, savename, text):
        """ Repeat augmentation. (e.g. "ワン" -> "ワンワン" (repeat_num=2))
        Args:
            repeat_num (int): repeat number
            label (str): label
            basename (str): basename
            savename (str): savename
            text (str): text
        Returns:
            int: mel frame length
        """
        duration, energy, kurtosis, mel, image, width = self._exist_feature_load(label, basename)
        dur_rep, ene_rep = np.tile(duration, repeat_num), np.tile(energy, repeat_num)
        kur_rep, wid_rep = np.tile(kurtosis, repeat_num), np.tile(width, repeat_num)
        mel_rep = np.tile(mel, (1, repeat_num))
        text_rep = text*repeat_num
        im_list = [image for _ in range(repeat_num)]
        # process visual onomatopoeia
        _im = im_list.pop(0)
        for im in im_list:
            dst = Image.new("RGB", (im.width+_im.width, im.height))
            dst.paste(_im, (0, 0))
            dst.paste(im, (_im.width, 0))
            _im = dst
        im_rep = dst
        # save
        np.save(self.path_preprocessed/"duration"/label/f"{savename}.npy", dur_rep)
        np.save(self.path_preprocessed/"energy"/label/f"{savename}.npy", ene_rep)
        np.save(self.path_preprocessed/"kurtosis"/label/f"{savename}.npy", kur_rep)
        np.save(self.path_preprocessed/"mel"/label/f"{savename}.npy", mel_rep.T)
        im_rep.save(self.path_preprocessed/"image"/"png"/label/f"{savename}.png")
        np.save(self.path_preprocessed/"image"/"width"/label/f"{savename}.npy", wid_rep)
        info = f"{savename}|{label}|{self.im_fontsize}|{self.path_font.stem}|{text_rep}"
        if self._is_traindata(savename):
            with open(os.path.join(self.path_preprocessed, "intermediate", "info", "train", label, f"{savename}.txt"), "w") as f:
                f.write(info)
        else:
            with open(os.path.join(self.path_preprocessed, "intermediate", "info", "val_test", label, f"{savename}.txt"), "w") as f:
                f.write(info)
        return mel_rep.shape[1]

    def _consecutiveaug(self, consecutive_num, pos, label, basename, savename, text):
        """ Consecutive augmentation. (e.g. "ワンワン" -> "ワワンワン" (consecutive_num=1, text_pos=0))
        Args:
            consecutive_num (int): consecutive number
            pos (int): position of target character
            label (str): label
            basename (str): basename
            savename (str): savename
            text (str): text
        Returns:
            int: mel frame length
            str: text after augmentation
        """
        duration, energy, kurtosis, mel, image, width = self._exist_feature_load(label, basename)
        def rep(values, pos, num):
            a = [values[pos]]*num
            return np.insert(values, pos, a)
        dur_rep, ene_rep = rep(duration, pos, consecutive_num-1), rep(energy, pos, consecutive_num-1)
        kur_rep, wid_rep = rep(kurtosis, pos, consecutive_num-1), rep(width, pos, consecutive_num-1)
        text_rep = text[:pos]+text[pos]*consecutive_num+text[pos+1:]
        # mel
        seg = mel[:, sum(duration[:pos]): sum(duration[:pos+1])]
        seg = np.tile(seg, (1, consecutive_num))
        mel_rep = np.insert(mel, [sum(duration[:pos])], seg, axis=1)
        seg = image.crop((sum(width[:pos]), 0, sum(width[:pos+1]), image.height))
        seg_list = [seg for _ in range(consecutive_num)]
        # process visual onomatopoeia
        im_left = image.crop((0, 0, sum(width[:pos]), image.height))
        im_right = image.crop((sum(width[:pos+1]), 0, image.width, image.height))
        im_list = [im_left]+seg_list+[im_right]
        _im = im_list.pop(0)
        for im in im_list:
            dst = Image.new("RGB", (im.width+_im.width, im.height))
            dst.paste(_im, (0, 0))
            dst.paste(im, (_im.width, 0))
            _im = dst
        im_rep = dst
        # save
        np.save(self.path_preprocessed/"duration"/label/f"{savename}.npy", dur_rep)
        np.save(self.path_preprocessed/"energy"/label/f"{savename}.npy", ene_rep)
        np.save(self.path_preprocessed/"kurtosis"/label/f"{savename}.npy", kur_rep)
        np.save(self.path_preprocessed/"mel"/label/f"{savename}.npy", mel_rep.T)
        im_rep.save(self.path_preprocessed/"image"/"png"/label/f"{savename}.png")
        np.save(self.path_preprocessed/"image"/"width"/label/f"{savename}.npy", wid_rep)
        info = f"{savename}|{label}|{self.im_fontsize}|{self.path_font.stem}|{text_rep}"
        if self._is_traindata(savename):
            with open(os.path.join(self.path_preprocessed, "intermediate", "info", "train", label, f"{savename}.txt"), "w") as f:
                f.write(info)
        else:
            with open(os.path.join(self.path_preprocessed, "intermediate", "info", "val_test", label, f"{savename}.txt"), "w") as f:
                f.write(info)
        return mel_rep.shape[1], text_rep
    
    def _get_consecutive_pos(self, text):
        """ Get consecutive position.
        Args:
            text (str): text
        Returns:
            int: position. If not exist, return None
        """
        pre_char = ""
        s_i, e_i = -1, -1
        consecutive_cnt = 1
        for i, char in enumerate(text):
            if char == pre_char:
                s_i = i-1 if consecutive_cnt==1 else s_i
                consecutive_cnt += 1
            else:
                if consecutive_cnt >= 3:
                    e_i = i-1
                    return s_i + int((e_i-s_i)/2)
                else:
                    s_i, e_i = -1, -1
                    consecutive_cnt = 1
                    pre_char = char
        if consecutive_cnt >= 3:
            e_i = len(text)-1
            return s_i + int((e_i-s_i)/2)
        return None
    
    def _normalize(self, in_dir, mean, std):
        """ Normalize.
        Args:
            in_dir (str or Path): input directory
            mean (float): mean
            std (float): standard deviation
        Returns:
            float: min value
            float: max value
        """
        max_value = np.finfo(np.float64).min
        min_value = np.finfo(np.float64).max
        for label in self.labels:
            folder_path = Path(os.path.join(in_dir, label))
            for filename in folder_path.glob("*.npy"):
                values = (np.load(filename) - mean) / std
                np.save(filename, values)

                max_value = max(max_value, max(values))
                min_value = min(min_value, min(values))

        return min_value, max_value
    
    def _remove_outlier(self, feature):
        """ Remove outlier. (IQR)
        Args:
            feature (np.ndarray): feature
        Returns:
            np.ndarray: feature without outlier
        """
        values = np.array(feature)
        p25 = np.percentile(values, 25)
        p75 = np.percentile(values, 75)
        lower = p25 - 1.5 * (p75 - p25)
        upper = p75 + 1.5 * (p75 - p25)
        normal_indices = np.logical_and(values > lower, values < upper)
        return values[normal_indices]
    
