# [ICASSP'23] Visual onoma-to-wave: Environmental sound synthesis from visually represented onomatopoeia
Official implementation of [Visual onoma-to-wave: environmental sound synthesis from visual onomatopoeias and sound-source images](https://arxiv.org/abs/2210.09173) ( to appear *ICASSP 2023* ) .

## Demo
[[Audio samples]](https://sarulab-speech.github.io/demo_visual-onoma-to-wave/)



## Quick start
Codes and pre-trained models will be available soon.

## Getting started
### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Download dataset
1. obtain [the onomatopoeia dataset for RWCP-SSD](https://github.com/KeisukeImoto/RWCPSSD_Onomatopoeia). It includes onomatopoeia words, and alignments.
```bash
cd RWCPSSD_Onomatopoeia
git submodule init
git submodule update
cd ..
```  
2. download [the RWCP-SSD dataset](https://staff.aist.go.jp/m.goto/RWCP-SSD/eng/index.html). It includes environmental sounds. After downloading, please put the dataset in `./corpus/rwcp-ssd/RWCP-SSD_Vol1/*`.

### 3. 3-step preprocessing
1. formatting the datase
- `config/preprocess.yaml`: Configuration file for formatting the dataset. Please edit the file according to your environment.
- `corpus/rwcp-ssd/RWCP-SSD_Vol1`: Path to the RWCP-SSD dataset.
```bash
python scripts/01_format.py config/preprocess.yaml corpus/rwcp-ssd/RWCP-SSD_Vol1
```  
2. lab file to TextGrid file
```bash
python scripts/02_prepare_tg.py config/preprocess.yaml
```
After this step, the following folder structure is created.
```
├── visual-onoma-to-wave
│   └── formatted_data/RWCP-SSD
│       ├── audio (wav files)
│       ├── lab
│       ├── text (onomatopoeia words)
│       └── TextGrid
```



## Code contributors
- Hien Ohnaka (National Institute of Technology, Tokuyama College, Japan.)
- [Shinnosuke Takamichi](https://sites.google.com/site/shinnosuketakamichi/home) (The University of Tokyo, Japan.)

## Reference
- [Onoma-to-wave: Environmental sound synthesis from onomatopoeic words](https://arxiv.org/abs/2102.05872)
- [HiFi-GAN: Generative Adversarial Networks for Efficient and High Fidelity Speech Synthesis](https://arxiv.org/abs/2010.05646)
- [FastSpeech2 implementation](https://github.com/Wataru-Nakata/FastSpeech2-JSUT)
  - Part of our codes follows this implementation.