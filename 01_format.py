import argparse
import os
import shutil
from pathlib import Path
import yaml
import pydub
import tqdm

def _format_rwcpssd(in_dir, out_dir):
    in_text_dir = Path("RWCPSSD_Onomatopoeia") / "RWCP_SSD_Onomatopoeia_jp" / "nospeech" / "drysrc"
    out_text_dir = Path(out_dir) / "text"
    in_lab_dir = Path("RWCPSSD_Onomatopoeia") / "RWCP_SSD_Onomatopoeia_jp_lab" / "nospeech" / "drysrc"
    out_lab_dir = Path(out_dir) / "lab"

    in_audio_dir = in_dir / "nospeech" / "drysrc"
    out_audio_dir = Path(out_dir) / "audio"

    def _normalize_text(text):
        for x in [str(os.sep), str(os.altsep), "|", "_"]:
            text = text.replace(x, "-")
        return text

    # organize onomatopoeia, audio and lab
    out_texts = []
    for in_text_path in sorted(in_text_dir.glob("**/*.ono")):
        if in_text_path.stem.startswith("."):
            continue
        
        in_base = in_text_path.relative_to(in_text_dir) # extract path from in_text_dir
        in_audio_path = in_audio_dir / in_base.parent / \
            "48khz" / in_base.with_suffix(".raw").name
        basename = _normalize_text(str(in_base.with_suffix("")))

        # audio event label
        event_label = in_text_path.parent.stem

        # audio
        out_audio_path = out_audio_dir / event_label / f"{basename}.wav"
        out_audio_path.parent.mkdir(parents=True, exist_ok=True)
        wav = pydub.AudioSegment.from_file(
            in_audio_path,
            format="raw",
            frame_rate=48000,
            channels=1,
            sample_width=2
        )
        wav.export(out_audio_path, format="wav")

        for worker_id, onomatopoeia_id, onomatopoeia, self_score in [x.split(",") for x in open(in_text_path, "r").readlines()]:
            onomatopoeia_id_raw = onomatopoeia_id
            onomatopoeia_id, onomatopoeia, self_score = _normalize_text(onomatopoeia_id), _normalize_text(onomatopoeia), self_score.replace('\n', '')
            acc_path = in_text_path.parent / f'{in_text_path.stem}.acc'

            # issue: particl2/071.acc does not exists.
            notexist_filepath = [
                "RWCPSSD_Onomatopoeia/RWCP_SSD_Onomatopoeia_jp/nospeech/drysrc/b1/particl2/071.acc"
            ]
            if str(acc_path) in notexist_filepath:
                continue
            
            # get average of other score. if self_score is 4 or 5, assign other score.
            # For details, please refer to the paper (-> https://dcase.community/documents/workshop2020/proceedings/DCASE2020Workshop_Okamoto_21.pdf).
            others_score = 0
            cnt = 0
            if int(self_score)>3 and (not str(acc_path) in notexist_filepath):
                for onomatopoeia_id_acc, onomatopoeia_acc, worker_id_for_assigned_other_score, other_score in [x.split(",") for x in open(acc_path, "r").readlines()]:
                    onomatopoeia_id_acc, onomatopoeia_acc, other_score = _normalize_text(onomatopoeia_id_acc), _normalize_text(onomatopoeia_acc), int(other_score.replace('\n', ''))
                    if onomatopoeia_id_acc == onomatopoeia_id:
                        others_score += other_score
                        cnt += 1
                cnt = 1 if cnt==0 else cnt
                others_score = others_score/cnt
            assert(others_score<=5), f"cnt: {cnt}, others_score: {others_score}, {acc_path}, {onomatopoeia}"

            out_texts.append([
                f"{basename}-{onomatopoeia_id}",
                out_audio_path.stem,
                onomatopoeia,
                event_label,
                str(self_score),
                str(others_score)
            ])

            # rename and replace lab file
            sound_id = in_text_path.stem
            in_lab_path = in_lab_dir / in_base.parent / f"{sound_id}-{onomatopoeia_id_raw}.lab"
            out_lab_path = out_lab_dir / event_label / f"{basename}-{onomatopoeia_id}.lab"
            out_lab_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(in_lab_path, out_lab_path)

    # write metadata in units of event_label
    for event_label in set(x[-3] for x in out_texts):
        ot = [x for x in out_texts if x[-3] == event_label]

        out_text_path = out_text_dir / event_label / "data.txt"
        out_text_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_text_path, "w", encoding="utf-8") as f:
            f.writelines(["|".join(t) + "\n" for t in ot])

def format_dataset(in_dir, out_dir, dataset="rwcp-ssd"):
    _format = {
        "rwcp-ssd": _format_rwcpssd
        # If you want to use a different dataset, you can add a formatter here.
    }

    if dataset in _format.keys():
        _format[dataset](in_dir, out_dir)
    else:
        raise ValueError(f"dataset {dataset} is not included in {_format}")


def load_args():
    parser = argparse.ArgumentParser(
        description="format dataset directory.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("preprocess_config_path", type=str,
                        help="file path of preprocess config. Please refer to the 'config/Readme.md' for details.")
    parser.add_argument("in_audio_dir", type=str,
                        help="Specify the path to the downloaded RWCP-SSD audio dataset.")
    parser.add_argument("--dataset", type=str, default=None,required=False,
                        help="specify formatter type. Currently, only rwcp-ssd is supported. If not, the dataset is specified in the config file.")
    
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = load_args()
    config = yaml.load(open(args.preprocess_config_path, "r"), Loader=yaml.FullLoader)
    in_dir = Path(args.in_audio_dir)
    out_dir = Path(config["path"]["formatted_data_path"])
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.dataset is None:
        dataset = config["dataset"]
    else:
        dataset = args.dataset
    format_dataset(in_dir, out_dir, dataset)
    print(f"data in {in_dir} is formatted and saved in {out_dir}.")
