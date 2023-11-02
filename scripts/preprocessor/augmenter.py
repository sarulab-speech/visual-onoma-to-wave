from pathlib import Path
from tqdm import tqdm
import numpy as np
import os
import json
from sklearn.preprocessing import StandardScaler
import torchaudio
import torch
from joblib import Parallel, delayed
import PIL
import random
import shutil

class Augmenter:
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
        self.im_bgcolor = config_dict["preprocessing"]["image"]["background_color"]
        self.im_txtcolor = config_dict["preprocessing"]["image"]["text_color"]
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

    def _get_basename(self, font_name, font_size, stem, ext=".png"):
        base = stem.replace(" ", "").replace("_", "-")
        return f"{font_name}_{font_size}pt_{base}{ext}"

    def _get_alignment(self, tier, non_trimwav, sil_margin_frame=0):
        sil_phones = ["sil", "sp", "spn", 'silB', 'silE', '']

        phones = []
        durations = np.array([])
        start_time = 0
        end_time = 0
        end_idx = 0
        wav_sec = len(non_trimwav)/self.sampling_rate
        start_np = np.array([])
        end_np = np.array([])
        sil_margin_sec = sil_margin_frame*self.hop_length/self.sampling_rate
        for t in tier._objects:
            s, e, p = t.start_time, t.end_time, t.text

            # Trim leading silences
            if phones == []:
                if p in sil_phones:
                    continue
                else:
                    start_time = s

            if p not in sil_phones:
                # For ordinary phones
                phones.append(p)
                end_time = e
                end_idx = len(phones)
            else:
                # For silent phones
                phones.append('sp')
                last_time = e

            start_np = np.append(start_np, s)
            end_np = np.append(end_np, e)

        durations = []
        wav2align_ratio = wav_sec/last_time
        start_time = start_time*wav2align_ratio
        end_time = end_time*wav2align_ratio
        last_time = last_time*wav2align_ratio
        start_np = start_np*wav2align_ratio
        end_np = end_np*wav2align_ratio
        # margin
        if start_time-sil_margin_sec < 0:
            start_time = 0
        else:
            start_time = start_time-sil_margin_sec
        start_np[0] = start_time
        if end_time+sil_margin_sec > last_time:
            end_time = last_time
        else:
            end_time = end_time+sil_margin_sec
        end_np[-2] = end_time
        for i in range(start_np.shape[0]):
            s = start_np[i]
            e = end_np[i]
            durations.append(
                int(
                    np.round(e * self.sampling_rate /
                             self.hop_length)
                    - np.round(s * self.sampling_rate /
                               self.hop_length)
                )
            )
        # Trim tailing silences
        phones = phones[:end_idx]
        durations = durations[:end_idx]
        assert len(phones) == len(durations), print(
            f'phones: {phones}\ndurations: {durations}')
        #print(f'phones: {phones}\ndurations: {durations}\nlast_time: {last_time}\nwav_sec:{wav_sec}')
        return phones, durations, start_time, end_time

    def _calc_spectrogram(self, audio):
        audio = torch.clip(torch.from_numpy(audio), -1, 1)
        magspec = self.spec_module(audio)
        melspec = self.mel_scale(magspec)
        logmelspec = torch.log(torch.clamp_min(
            melspec, 1.0e-5) * 1.0).to(torch.float32)
        energy = torch.norm(magspec, dim=0)
        return logmelspec.numpy(), energy.numpy()

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

    def _exist_feature_load(self, label, basename):
        duration = np.load(os.path.join(
            self.path_preprocessed, "duration", label, "{}.npy".format(basename)))
        energy = np.load(os.path.join(
            self.path_preprocessed, "energy", label, "{}.npy".format(basename)))
        mel_spectrogram = np.load(os.path.join(
            self.path_preprocessed, "mel", label, "{}.npy".format(basename))).T
        image = PIL.Image.open(os.path.join(
            self.path_preprocessed, "image", "png", label, f'{basename}.png'))
        width = np.load(os.path.join(
            self.path_preprocessed, "image", "width", label, "{}.npy".format(basename)))
        return duration, energy, mel_spectrogram, image, width

    def _get_concat_h_multi(self, im_list):
        def concat_h(im1, im2):
            dst = PIL.Image.new('RGB', (im1.width + im2.width, im1.height))
            dst.paste(im1, (0, 0))
            dst.paste(im2, (im1.width, 0))
            return dst
        _im = im_list.pop(0)
        for im in im_list:
            _im = concat_h(_im, im)
        return _im

    def _is_traindata(self, savename):
        audio_numbering = int(savename.split('-')[2])
        if audio_numbering not in self.p_untrained_dataid:
            return True
        else:
            return False

    def _augmentation_repeat(self, repeat_num, label, basename, savename, text):
        duration, energy, mel_spectrogram, image, width = self._exist_feature_load(label, basename)
        dur_rep, energy_rep, width_rep, mel_rep = \
            np.tile(duration, repeat_num), \
                np.tile(energy, repeat_num), \
                    np.tile(width, repeat_num), \
                        np.tile(mel_spectrogram, (1, repeat_num))
        im_list = [image for i in range(repeat_num)]
        image_rep = self._get_concat_h_multi(im_list)
        # save
        savename = f"{basename}_repeat{repeat_num}"
        np.save(os.path.join(self.path_preprocessed, "duration",
                label, f"{savename}.npy"), dur_rep)
        np.save(os.path.join(self.path_preprocessed, "energy",
                label, f"{savename}.npy"), energy_rep)
        np.save(os.path.join(self.path_preprocessed, "mel",
                label, f"{savename}.npy"), mel_rep.T)
        image_rep.save(os.path.join(self.path_preprocessed, "image", "png", label, f"{savename}.png"))
        np.save(os.path.join(self.path_preprocessed, "image", "width", label, f"{savename}.npy"), width_rep)
        info = f"{savename}|{label}|{self.im_fontsize}|{self.path_font.stem}|{text*repeat_num}"
        if self._is_traindata(savename):
            with open(os.path.join(self.path_preprocessed, "intermediate", "info", "train", label, f"{savename}.txt"), "w") as f:
                f.write(info)
        else:
            with open(os.path.join(self.path_preprocessed, "intermediate", "info", "val_test", label, f"{savename}.txt"), "w") as f:
                f.write(info)
        return mel_rep.shape[1], text*repeat_num

    def _augmentation_consecutive(self, consecutive_num, repeatbase_pos, label, basename, savename, text):
        duration, energy, mel_spectrogram, image, width = self._exist_feature_load(label, basename)
        def aug(values, pos, consecutive_num):
            rep = [values[pos]]*consecutive_num
            return np.insert(values, pos, rep)
        dur_rep, energy_rep, width_rep = \
            aug(duration, repeatbase_pos, consecutive_num), \
                aug(energy, repeatbase_pos, consecutive_num), \
                    aug(width, repeatbase_pos, consecutive_num)
        s_idx = sum(duration[:repeatbase_pos])
        segment = mel_spectrogram[:, s_idx:s_idx+duration[repeatbase_pos]]
        segment = np.tile(segment, (1, consecutive_num))
        mel_rep = np.insert(mel_spectrogram, [s_idx], segment, axis=1)
        segment = image.crop((sum(width[:repeatbase_pos]), 0, sum(width[:repeatbase_pos+1]), self.im_fontsize))
        seg_list = [segment for i in range(consecutive_num+1)]
        image_left = image.crop((0, 0, sum(width[:repeatbase_pos]), self.im_fontsize))
        image_right = image.crop((sum(width[:repeatbase_pos+1]), 0, image.width, self.im_fontsize))
        image_rep = self._get_concat_h_multi([image_left]+seg_list+[image_right])
        np.save(os.path.join(self.path_preprocessed, "duration",
                label, f"{savename}.npy"), dur_rep)
        np.save(os.path.join(self.path_preprocessed, "energy",
                label, f"{savename}.npy"), energy_rep)
        np.save(os.path.join(self.path_preprocessed, "mel",
                label, f"{savename}.npy"), mel_rep.T)
        image_rep.save(os.path.join(self.path_preprocessed, "image", "png", label, f"{savename}.png"))
        np.save(os.path.join(self.path_preprocessed, "image", "width", label, f"{savename}.npy"), width_rep)
        base_chara = text[repeatbase_pos]
        base_chara = base_chara*consecutive_num
        text_consecutive = text[:repeatbase_pos+1] + base_chara + text[repeatbase_pos+1:]
        info = f"{savename}|{label}|{self.im_fontsize}|{self.path_font.stem}|{text_consecutive}"
        if self._is_traindata(savename):
            with open(os.path.join(self.path_preprocessed, "intermediate", "info", "train", label, f"{savename}.txt"), "w") as f:
                f.write(info)
        else:
            with open(os.path.join(self.path_preprocessed, "intermediate", "info", "val_test", label, f"{savename}.txt"), "w") as f:
                f.write(info)
        return mel_rep.shape[1], text_consecutive

    def _get_consecutive_pos(self, text):
        pre_char = ""
        s_i, e_i = -1, -1
        consecutive_cnt = 1
        for i, char in enumerate(text):
            if char == pre_char:
                s_i = i-1 if consecutive_cnt==1 else s_i
                consecutive_cnt += 1
            else:
                if consecutive_cnt >= 3:
                    return s_i + int((e_i-s_i)/2)
                else:
                    s_i, e_i = -1, -1
                    consecutive_cnt = 1
                    pre_char = char
        if consecutive_cnt >= 3:
            return s_i + int((e_i-s_i)/2)
        return None

    def _process(self, label, line):
        text_base, audio_base, text, _, confidence_score, acceptance_score = line.replace('\n','').split("|")
        basename = self._get_basename(self.path_font.stem, self.im_fontsize, text_base, ext="")
        aug_n_sum = 0
        # augmentation of repeated text
        repeat_num = 2
        while (repeat_num <= self.aug_repeatnum) and (len(text) <= self.aug_maxlen):
            savename = f"{basename}_repeat{repeat_num}"
            n, _ = self._augmentation_repeat(repeat_num, label, basename, savename, text)
            repeat_num += 1
            aug_n_sum += n
        # augmentation of repeating first character (not included ICASSP2023 paper: In default, self.aug_first_consecutive=0)
        augment_firstc = 1
        while (augment_firstc <= self.aug_first_consecutive):
            savename = f"{basename}_first{augment_firstc}"
            n, _ = self._augmentation_consecutive(augment_firstc, 0, label, basename, savename, text)
            augment_firstc += 1
            aug_n_sum += n
        # augmentation of repeating consecutive characters
        repeat_base_pos = self._get_consecutive_pos(text)
        if repeat_base_pos is not None:
            augment_consecutive = 1
            while (augment_consecutive <= self.aug_chara_consecutive_num):
                savename = f"{basename}_continue{augment_consecutive}"
                n, ret_text = self._augmentation_consecutive(augment_consecutive, repeat_base_pos, label, basename, savename, text)
                augment_consecutive += 1
                aug_n_sum += n
                repeat_num = 2
                basename_ret = savename
                while (repeat_num <= self.aug_repeatnum) and (len(ret_text) <= self.aug_maxlen):
                    savename = f"{basename}_repeat{repeat_num}_continue{augment_consecutive-1}"
                    n, _ = self._augmentation_repeat(repeat_num, label, basename_ret, savename, ret_text)
                    repeat_num += 1
                    aug_n_sum += n
        return aug_n_sum

    def _process_visual_ono(self, label, info, wav_len, Visual_ono_Generator):
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

    def _remove_outlier(self, energy):
        values = np.array(energy)
        p25 = np.percentile(values, 25)
        p75 = np.percentile(values, 75)
        lower = p25 - 1.5 * (p75 - p25)
        upper = p75 + 1.5 * (p75 - p25)
        normal_indices = np.logical_and(values > lower, values < upper)
        return values[normal_indices]
    
    def _normalize(self, in_dir, mean, std):
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

    def build_from_path(self, info_list_list, n_frames_cnt, num_workers=10):
        if len(self.p_uselabel) != 0:
            self.labels = sorted(list(
                set([x.parent.stem for x in (self.path_formatted / "audio").glob("*/*.wav")]) &
                set(self.p_uselabel)
            ))
        else: # use all labels
            self.labels = sorted(list(
                set([x.parent.stem for x in (self.path_formatted / "audio").glob("*/*.wav")])
            ))
        outer_bar = tqdm(total=len(self.labels), desc="Sound Label", position=0)
        # compute mel-spectrogram, duration, energy
        for label, info_list in zip(self.labels, info_list_list):
            # joblib
            results = Parallel(n_jobs=num_workers, verbose=5)(
                delayed(self._process)(label, line) for line in info_list
            )
            mel_lens_aug = np.array(results)
            n_frames_cnt += np.sum(mel_lens_aug)
            outer_bar.update(1)
        print("===energy normalization===")
        energy_scaler = StandardScaler()
        energy_npy_list = list(Path(self.path_preprocessed / "energy").glob("*/*.npy"))
        bar = tqdm(total=len(energy_npy_list), desc="Energy", position=0)
        for energy_npy in energy_npy_list:
            energy = np.load(energy_npy)
            energy = self._remove_outlier(energy)
            if energy.size == 0:
                bar.update(1)
                continue
            energy = energy.reshape(-1, 1)
            energy_scaler.partial_fit(energy)
            bar.update(1)
        energy_mean = energy_scaler.mean_[0] if self.energy_normalization else 0
        energy_std = energy_scaler.scale_[0] if self.energy_normalization else 1
        energy_min, energy_max = self._normalize(self.path_preprocessed / "energy", energy_mean, energy_std)
        with open(os.path.join(self.path_preprocessed, "stats.json"), "w") as f:
            stats = {
                "energy": [
                    float(energy_min),
                    float(energy_max),
                    float(energy_mean),
                    float(energy_std)
                ]
            }
            f.write(json.dumps(stats))
        print("===write metadata===")
        train_info_list = list(Path(self.path_preprocessed / "intermediate" / "info" / "train").glob("*/*.txt"))
        random.shuffle(train_info_list)
        with open(os.path.join(self.path_preprocessed, "train.txt"), "w") as f:
            for train_info in train_info_list:
                f.write(train_info.read_text())
                f.write("\n")
        val_test_info_list = list(Path(self.path_preprocessed / "intermediate" / "info" / "val_test").glob("*/*.txt"))
        random.shuffle(val_test_info_list)
        val_list, test_list = val_test_info_list[:len(val_test_info_list)//2], val_test_info_list[len(val_test_info_list)//2:]
        with open(os.path.join(self.path_preprocessed, "val.txt"), "w") as f:
            for val_info in val_list:
                f.write(val_info.read_text())
                f.write("\n")
        with open(os.path.join(self.path_preprocessed, "test.txt"), "w") as f:
            for test_info in test_list:
                f.write(test_info.read_text())
                f.write("\n")
        shutil.rmtree(self.path_preprocessed / "intermediate")
        print("===Done===")
        print(
            "Total time: {} hours".format(
                n_frames_cnt * self.hop_length / self.sampling_rate / 3600
            )
        )
