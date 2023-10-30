import argparse
import pathlib
from pathlib import Path
from tqdm import tqdm
from convert_label import read_lab
import numpy as np
import scipy.stats as stats
import os
import json
import yaml

if __name__ == '__main__':
    """ Create TextGrid files from lab files. """
    parser = argparse.ArgumentParser()
    parser.add_argument("config_path", type=str,
                        help="filename of preprocess yaml file.")
    args = parser.parse_args()
    config = yaml.load(open(args.config_path, "r"), Loader=yaml.FullLoader)
    lab_dir = Path(config["path"]["formatted_data_path"]) / "lab"
    tg_dir = Path(config["path"]["formatted_data_path"]) / "TextGrid"

    audio_labels = list(
      set([x.parent.stem for x in (lab_dir).glob("*/*.lab")])
    )


    # create output directory
    label_length_param = {}
    maximum_ = -1
    minimum_ = 100
    for i,audio_label in tqdm(enumerate(audio_labels)):
        tg_kata_dir = (tg_dir / audio_label)
        if not tg_kata_dir.exists():
            tg_kata_dir.mkdir(parents=True)

        # iter through lab files
        audio_level_length = np.array([])
        for lab_file in (lab_dir / audio_label).glob("*.lab"):
            label, chara_len = read_lab(str(lab_file), 'katakana')
            audio_level_length = np.append(audio_level_length, chara_len)
            textgridFilePath = tg_kata_dir/lab_file.with_suffix('.TextGrid').name
            label.to_textgrid(textgridFilePath)
        mean_ = np.mean(audio_level_length)
        median_ = np.median(audio_level_length)
        mode_, _ = stats.mode(audio_level_length)
        max_ = np.max(audio_level_length)
        min_ = np.min(audio_level_length)
        maximum_ = max(maximum_, max_)
        minimum_ = min(minimum_, min_)
        label_length_param[audio_label] = (mean_, median_, int(mode_), max_, min_)
    label_length_param["all param"] = (maximum_, minimum_)
    with open(os.path.join(config["path"]["formatted_data_path"], "dataset_length.json"), "w") as f:
        f.write(json.dumps(label_length_param))
