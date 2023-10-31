from pathlib import Path
from tqdm import tqdm
import librosa
import numpy as np
import os
import json
import tgt
from sklearn.preprocessing import StandardScaler
import torchaudio
import torch
from joblib import Parallel, delayed
from .visualtext_generator import Generator

class Preprocessor:
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

    def _makedirs(self, label):
        _path = self.path_preprocessed / "duration" / label
        _path.mkdir(parents=True, exist_ok=True)
        _path = self.path_preprocessed / "energy" / label
        _path.mkdir(parents=True, exist_ok=True)
        _path = self.path_preprocessed / "mel" / label
        _path.mkdir(parents=True, exist_ok=True)
        _path = self.path_preprocessed / "image" / "png" / label
        _path.mkdir(parents=True, exist_ok=True)
        _path = self.path_preprocessed / "image" / "width" / label
        _path.mkdir(parents=True, exist_ok=True)

    def _check_score_border(self, confidence_score, acceptance_score):
        if float(confidence_score) < self.p_confidence_score_border:
            return False
        if float(acceptance_score) < self.p_acceptance_score_border:
            return False
        return True

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

    def _process(self, label, line):
        text_base, audio_base, text, _, confidence_score, acceptance_score = line.replace('\n','').split("|")
        # check score border
        if not self._check_score_border(confidence_score, acceptance_score):
            return -1,-1,-1
        tg_path = os.path.join(
            self.path_formatted, "TextGrid", label, "{}.TextGrid".format(
                text_base)
        )
        basename = self._get_basename(self.path_font.stem, self.im_fontsize, text_base, ext="")
        if not os.path.exists(tg_path):
            return -1,-1,-1
        wav_path = os.path.join(self.path_formatted, "audio",
                                label, "{}.wav".format(audio_base))
        wav, _ = librosa.load(wav_path, sr=22050)
        textgrid = tgt.io.read_textgrid(tg_path)
        character, duration, start, end = self._get_alignment(
            textgrid.get_tier_by_name("phones"),
            wav,
            sil_margin_frame=self.margin_frame,
        )
        if len(character) != len(text):
            return -1,-1,-1
        trimed_wav = wav[
            int(self.sampling_rate * start):
        ].astype(np.float32)
        wav_len = len(trimed_wav)
        if start >= end:
            return -1,-1,-1
        if len(wav[int(self.sampling_rate * start): int(self.sampling_rate * end)]) < len(wav)/10:
            return -1,-1,-1
        mel_spectrogram, energy = self._calc_spectrogram(trimed_wav)
        mel_spectrogram = mel_spectrogram[:, : sum(duration)]
        energy = energy[: sum(duration)]
        if self.energy_character_averaging:
            pos = 0
            for i, d in enumerate(duration):
                if d > 0:
                    energy[i] = np.mean(energy[pos: pos + d])
                else:
                    energy[i] = 0
                pos += d
            energy = energy[: len(duration)]
        filename = "{}.npy".format(basename)
        np.save(os.path.join(self.path_preprocessed, "duration",
                label, filename), duration)
        # save energy
        np.save(os.path.join(self.path_preprocessed, "energy",
                label, filename), energy)
        # save mel
        np.save(os.path.join(self.path_preprocessed, "mel",
                label, filename), mel_spectrogram.T)
        return mel_spectrogram.shape[1], wav_len, len(text)

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

    def build_from_path(self, num_workers=10):
        if len(self.p_uselabel) != 0:
            self.labels = sorted(list(
                set([x.parent.stem for x in (self.path_formatted / "audio").glob("*/*.wav")]) &
                set(self.p_uselabel)
            ))
        else: # use all labels
            self.labels = sorted(list(
                set([x.parent.stem for x in (self.path_formatted / "audio").glob("*/*.wav")])
            ))       
        audio_labels = {}
        width_dumps = {} # {label: character_len_per1sec}
        info_list_list = []
        wav_lens_list = []
        n_frames_cnt = 0
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
        for i, info_list, wav_lens in zip(range(len(info_list_list)), info_list_list, wav_lens_list):
            label = [k for k, v in audio_labels.items() if v == i][0]
            character_persec, max_width, _ = width_dumps[label]
            Visual_ono_Generator = Generator(self.config, character_persec, max_width)
            Parallel(n_jobs=num_workers, verbose=1)(
                delayed(self._process_visual_ono)(label, info, wav_len, Visual_ono_Generator) for info, wav_len in zip(info_list, wav_lens)
            )