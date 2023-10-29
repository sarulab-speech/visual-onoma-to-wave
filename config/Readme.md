# What is each config.yaml?
## preprocess.yaml
- dataset: Dataset to be used. (only support `rwcp-ssd`)

- path
    - corpus_data_path: path of corpus data.
    - raw_data_path: After running step2, path of processed data folder path.
    - preprocessed_data_path: After running step3, path of processed data folder path.
    - font_path: Path to font data for visual onomatopoeia creation.

- preprocessing:
    - **extract_label**: Specify which sound event to use.
    - notuse_train_audio_num: Sound id to be reserved for val and test. (`rwcp-ssd` has sound-id (0~99) for each sound event.)
    - confidence_score_border: Onomatopoeia data cleaning based on confidence score. (Please refer to [RWCP-SSD-Onomatopoeia](https://github.com/KeisukeImoto/RWCPSSD_Onomatopoeia) for details.)
    - acceptance_score_border: Onomatopoeia data cleaning based on acceptance score. (Please refer to [RWCP-SSD-Onomatopoeia](https://github.com/KeisukeImoto/RWCPSSD_Onomatopoeia) for details.)
    - text:
        - font_size: Font size of visual onomatopoeia.
    - augmentation:
        - augment_maxlen: Maximum character length of the original onomatopoeia for data augmentation.
        - augment_repeatnum: Maximum number of repetitions for word-level data augmentation.
        - chara_consecutive_num: Maximum number of repetitions for character-level data augmentation.
    - image:
        - load_scale: support "gray-scale" or "RGB-scale"
        - image_stretching: Specifies whether image stretching is used.
        - background_color: 3-dimension. (if selected "gray-scale", Converted to one dimension in the program. )
        - text_color: Same rules as background_color.
        - pad_color: Color of padding when adjusting width. (1 dimension. Same number as RGB)
    - audio:
        - sampling_rate: only support 22050.
        - max_wav_value: 
    - stft:
        - filter_length:
        - hop_length:
        - win_length:
        - margin_frame: 03_preprocessing's precision in silent cropping sometimes results in the cropping of a sound segment. `margin_frame` allows for an extra frame of grace in this cropping.
    - mel:
        - n_mel_channels: 
        - mel_fmin:
        - mel_fmax:
    energy:
        feature: 
        normalization:

## model.yaml
Same as [vTTS](https://github.com/Yoshifumi-Nakano/visual-text-to-speech) and [FastSpeech2-JSUT](https://github.com/Wataru-Nakata/FastSpeech2-JSUT)

## train.yaml
- path: 
    - ckpt_path: Path of trained model parameter.
    - log_path: Path of training log.
    - result_path: Path of evaluation & prediction result.
- optimizer & step:
    Same as [vTTS](https://github.com/Yoshifumi-Nakano/visual-text-to-speech) and [FastSpeech2-JSUT](https://github.com/Wataru-Nakata/FastSpeech2-JSUT) without `init-lr`.

- use_image: Specify whether to use image information (visual-text)
- image_encoder: Specifies whether the CLIP image-encoder is used for conditioning. Sorry, but it has not been tested to work with this open source implementation.