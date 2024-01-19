import os
import json
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib
matplotlib.use('Agg')
from scipy.io import wavfile
from matplotlib import pyplot as plt
plt.ioff()
import torchvision.transforms as transforms
import audio as Audio
import cv2
import io
# import clip
from PIL import Image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose(
    [transforms.ToTensor()]
)

def to_device(data, device):
    (
        ids,
        audiotypes,
        texts,
        src_lens,
        max_src_len,
        mels,
        mel_lens,
        max_mel_len,
        energies,
        durations,
        images,
        event_image_features
    ) = data

    audiotypes = torch.from_numpy(audiotypes).long().to(device)
    src_lens = torch.from_numpy(src_lens).to(device)
    if mels is not None:
        mels = torch.from_numpy(mels).float().to(device)
    if mel_lens is not None:
        mel_lens = torch.from_numpy(mel_lens).to(device)
    if energies is not None:
        energies = torch.from_numpy(energies).to(device)
    if durations is not None:
        durations = torch.from_numpy(durations).long().to(device)
    if images is not None:
        images=torch.stack([transform(im) for im in images]).to(device)
    if event_image_features[0] is not None:
        event_image_features = torch.from_numpy(event_image_features).to(device)
    else:
        event_image_features=None
    texts=torch.from_numpy(texts).long().to(device)

    return (
            ids,
            audiotypes,
            texts,
            src_lens,
            max_src_len,
            mels,
            mel_lens,
            max_mel_len,
            energies,
            durations,
            images,
            event_image_features
        )

def to_device_synth(data, device):
    (
        ids,
        audiotypes,
        texts,
        src_lens,
        max_src_len,
        mels,
        mel_lens,
        max_mel_len,
        energies,
        durations,
        images,
        event_image_features
    ) = data

    audiotypes = torch.from_numpy(audiotypes).long().to(device)
    src_lens = torch.from_numpy(src_lens).to(device)
    if mels is not None:
        mels = torch.from_numpy(mels).float().to(device)
    if mel_lens is not None:
        mel_lens = torch.from_numpy(mel_lens).to(device)
    if energies is not None:
        energies = torch.from_numpy(energies).to(device)
    if durations is not None:
        durations = torch.from_numpy(durations).long().to(device)
    if images is not None:
        images=torch.stack([transform(im) for im in images]).to(device)
    if event_image_features[0] is not None:
        event_image_features = torch.from_numpy(event_image_features).to(device)
    else:
        event_image_features=None
    texts=torch.from_numpy(texts).long().to(device)

    return ((
            ids,
            audiotypes,
            texts,
            src_lens,
            max_src_len,
            mels,
            mel_lens,
            max_mel_len,
            energies,
            durations,
            images,
            event_image_features
        ),
        (
            ids,
            audiotypes,
            texts,
            src_lens,
            max_src_len,
            None,
            None,
            None,
            None,
            None,
            images,
            event_image_features
        )
        )

def log(
    logger, step=None, losses=None, fig=None, audio=None, image=None, sampling_rate=22050, tag=""
):
    if losses is not None:
        logger.add_scalar("Loss/total_loss", losses[0], step)
        logger.add_scalar("Loss/mel_loss", losses[1], step)
        logger.add_scalar("Loss/mel_postnet_loss", losses[2], step)
        logger.add_scalar("Loss/energy_loss", losses[3], step)
        logger.add_scalar("Loss/duration_loss", losses[4], step)

    if fig is not None:
        logger.add_figure(tag, fig)

    if image is not None:
        logger.add_image(tag, image.transpose(2,0,1))

    if audio is not None:
        logger.add_audio(
            tag,
            audio / max(abs(audio)),
            sample_rate=sampling_rate,
        )

def get_mask_from_lengths(lengths, max_len=None):
    batch_size = lengths.shape[0]
    if max_len is None:
        max_len = torch.max(lengths).item()

    ids = torch.arange(0, max_len).unsqueeze(0).expand(batch_size, -1).to(device)
    mask = ids >= lengths.unsqueeze(1).expand(-1, max_len)

    return mask

def expand(values, durations):
    out = list()
    for value, d in zip(values, durations):
        out += [value] * max(0, int(d))
    return np.array(out)

def synth_one_sample(targets, predictions, vocoder, model_config, preprocess_config, use_image=True):
    basename = targets[0][0]
    data_name = targets[0][0].split("_")[-1]
    preprocessed_path = preprocess_config["path"]["preprocessed_data_path"]
    with open(os.path.join(preprocessed_path, "audiotype.json")) as f:
        audio_map = json.load(f)
        label = [k for k,v in audio_map.items() if v==targets[1][0].item()][0]
    # prepare prediction's data
    src_len = predictions[7][0].item()
    mel_len = predictions[8][0].item()
    mel_prediction = predictions[1][0, :mel_len].detach().transpose(0, 1)
    e_prediction = predictions[2][0]
    d_prediction = predictions[4][0]
    # prepare target's data
    mel_target = targets[5][0, :mel_len].detach().transpose(0, 1)
    duration = targets[9][0, :src_len].detach().cpu().numpy()
    energy_break = [duration[0]]
    for i in range(1,len(duration)-1):
        energy_break.append(energy_break[i-1]+duration[i])
    # load image
    if use_image:
        assess_p = os.path.join(preprocessed_path, "image_assessment", label, f"{basename}.png")
        if os.path.exists(assess_p):
            input_img = cv2.imread(assess_p, cv2.IMREAD_GRAYSCALE)
        else:
            image_p = os.path.join(preprocessed_path, "image", "png", label, f"{basename}.png")
            input_img = cv2.imread(image_p, cv2.IMREAD_GRAYSCALE)
    
    if preprocess_config["preprocessing"]["energy"]["feature"] == "element_level":
        energy = targets[8][0, :src_len].detach().cpu().numpy()
        energy = expand(energy, duration)
    else:
        energy = targets[8][0, :mel_len].detach().cpu().numpy()

    with open(
        os.path.join(preprocess_config["path"]["preprocessed_data_path"], "stats.json")
    ) as f:
        stats = json.load(f)
        stats = stats["energy"][:2]
    if use_image:
        im = plot_mel_withinput(
            input_img,
            [
                (mel_prediction.cpu().numpy(), energy, energy_break),
                (mel_target.cpu().numpy(), energy, energy_break)
            ],
            stats,
            [data_name, "Synthetized\nSpectrogram", "Ground-Truth\nSpectrogram"]
        )
    else:
        im = plot_mel(
            [
                (mel_prediction.cpu().numpy(), energy, energy_break),
                (mel_target.cpu().numpy(), energy, energy_break)
            ],
            stats,
            ["Synthetized\nSpectrogram", "Ground-Truth\nSpectrogram"]
        )

    if vocoder is not None:
        from .model import vocoder_infer

        wav_reconstruction = vocoder_infer(
            mel_target.unsqueeze(0),
            vocoder,
            model_config,
            preprocess_config,
            Normalize=False
        )[0]
        wav_prediction = vocoder_infer(
            mel_prediction.unsqueeze(0),
            vocoder,
            model_config,
            preprocess_config,
            Normalize=False
        )[0]

    return im, wav_reconstruction, wav_prediction, basename


def synth_samples(targets, predictions, vocoder, model_config, preprocess_config, path, show_target=True, image_assessment=None, event_img_stem=''):
    basenames = targets[0]
    for i in range(len(targets[0])):
        basename = f'{basenames[i]}_{event_img_stem}'
        data_name = basenames[i].split("_")[-1]
        src_len = predictions[7][i].item()
        mel_len = predictions[8][i].item()
        if image_assessment is None:
            input_img = targets[10][i, :src_len*24].permute(1,2,0).detach().cpu().numpy()
        else:
            input_img = image_assessment

        # prediction
        mel_prediction = predictions[1][i, :mel_len].detach().transpose(0, 1)
        duration = predictions[4][i, :src_len].detach().cpu().numpy()
        energy_break = [duration[0]]
        for j in range(1,len(duration)-1):
            energy_break.append(energy_break[j-1]+duration[j])    
        if preprocess_config["preprocessing"]["energy"]["feature"] == "element_level":
            energy = predictions[2][i, :src_len].detach().cpu().numpy()
            energy = expand(energy, duration)
        else:
            energy = predictions[2][i, :mel_len].detach().cpu().numpy()
        
        if show_target:
            # target
            mel_len_gt = targets[6][i]
            mel_target = targets[5][i, :mel_len_gt].detach().transpose(0, 1)
            duration_target = targets[9][i, :src_len].detach().cpu().numpy()
            energy_break_target = [duration_target[0]]
            for a in range(1,len(duration_target)-1):
                energy_break_target.append(energy_break_target[a-1]+duration_target[a])
            if preprocess_config["preprocessing"]["energy"]["feature"] == "element_level":
                energy_target = targets[8][i, :src_len].detach().cpu().numpy()
                energy_target = expand(energy_target, duration_target)
            else:
                energy_target = targets[8][i, :mel_len_gt].detach().cpu().numpy()

        with open(
            os.path.join(preprocess_config["path"]["preprocessed_data_path"], "stats.json")
        ) as f:
            stats = json.load(f)
            stats = stats["energy"][:2]

        if show_target:
            im = plot_mel_withinput(
                input_img,
                [
                    (mel_prediction.cpu().numpy(), energy, energy_break),
                    (mel_target.cpu().numpy(), energy_target, energy_break_target)
                ],
                stats,
                [data_name, "Synthetized\nSpectrogram", "Ground-Truth\nSpectrogram"],
            )
        else:
            with open( os.path.join(preprocess_config["path"]["preprocessed_data_path"], "audiotype.json")) as f:
                audio_map = json.load(f)
                eventlabel = [k for k,v in audio_map.items() if v==targets[1][0]][0]
            im = plot_mel_withinput(
                input_img,
                [
                    (mel_prediction.cpu().numpy(), energy, energy_break)
                ],
                stats,
                [f"event_label: {eventlabel}", "Synthetized\nSpectrogram"],
            )
        
        cv2.imwrite(os.path.join(path, "{}.png".format(basename)), im)

    from .model import vocoder_infer

    mel_predictions = predictions[1].transpose(1, 2)
    lengths = predictions[8] * preprocess_config["preprocessing"]["stft"]["hop_length"]
    wav_predictions = vocoder_infer(
        mel_predictions, vocoder, model_config, preprocess_config, lengths=lengths
    )
    STFT = Audio.stft.TacotronSTFT(
      preprocess_config["preprocessing"]["stft"]["filter_length"],
      preprocess_config["preprocessing"]["stft"]["hop_length"],
      preprocess_config["preprocessing"]["stft"]["win_length"],
      preprocess_config["preprocessing"]["mel"]["n_mel_channels"],
      preprocess_config["preprocessing"]["audio"]["sampling_rate"],
      preprocess_config["preprocessing"]["mel"]["mel_fmin"],
      preprocess_config["preprocessing"]["mel"]["mel_fmax"],
    )
    # wav_predict_gl = Audio.tools.inv_mel_spec(mel_prediction, f"{basename}.wav", STFT, 500)
    
    if show_target:
        mel_targets = targets[5].transpose(1,2)
        wav_reconstructions = vocoder_infer(
                mel_targets,
                vocoder,
                model_config,
                preprocess_config,
                lengths=targets[6] * preprocess_config["preprocessing"]["stft"]["hop_length"]
        )

    sampling_rate = preprocess_config["preprocessing"]["audio"]["sampling_rate"]
    if show_target:
        for wavp, wavr, basename in zip(wav_predictions, wav_reconstructions, basenames):
            basename = f'{basename}_{event_img_stem}'
            print(os.path.join(path, "{}.wav".format(basename)))
            wavfile.write(os.path.join(path, "{}_synthesis.wav".format(basename)), sampling_rate, wavp)
            wavfile.write(os.path.join(path, "{}_reconst.wav".format(basename)), sampling_rate, wavr)
    else:
        for wavp, basename in zip(wav_predictions, basenames):
            print(os.path.join(path, "{}.wav".format(basename)))
            wavfile.write(os.path.join(path, "{}_synthesis.wav".format(basename)), sampling_rate, wavp)

def synth_for_eval(targets, predictions, vocoder, model_config, preprocess_config, synth_savepath, save_reconst=False, reconst_savepath=None):
    basenames = targets[0]
    id2text = {
        0:"ベルを鳴らす音",
        1:"目覚まし時計の音",
        2:"コーヒー豆を手動ミルで挽く音",
        3:"カップを叩く音",
        4:"ドラムを叩く音",
        5:"マラカスの音",
        6:"ひげ剃りの動作音",
        7:"紙を引き裂く音",
        8:"金属製のゴミ箱を叩く音",
        9:"笛の音"
    }

    from .model import vocoder_infer

    mel_predictions = predictions[1].transpose(1, 2)
    lengths = predictions[8] * preprocess_config["preprocessing"]["stft"]["hop_length"]
    wav_predictions = vocoder_infer(
        mel_predictions, vocoder, model_config, preprocess_config, lengths=lengths
    )
    STFT = Audio.stft.TacotronSTFT(
      preprocess_config["preprocessing"]["stft"]["filter_length"],
      preprocess_config["preprocessing"]["stft"]["hop_length"],
      preprocess_config["preprocessing"]["stft"]["win_length"],
      preprocess_config["preprocessing"]["mel"]["n_mel_channels"],
      preprocess_config["preprocessing"]["audio"]["sampling_rate"],
      preprocess_config["preprocessing"]["mel"]["mel_fmin"],
      preprocess_config["preprocessing"]["mel"]["mel_fmax"],
    )
    if save_reconst:
        mel_targets = targets[5].transpose(1,2)
        wav_reconstructions = vocoder_infer(
                mel_targets,
                vocoder,
                model_config,
                preprocess_config,
                lengths=targets[6] * preprocess_config["preprocessing"]["stft"]["hop_length"]
        )
        sampling_rate = preprocess_config["preprocessing"]["audio"]["sampling_rate"]
        for wavp, wavr, basename in zip(wav_predictions, wav_reconstructions, basenames):
            savename = "_".join(basename.split("_")[:-1])
            wavfile.write(os.path.join(synth_savepath, "{}.wav".format(savename)), sampling_rate, wavp)
            wavfile.write(os.path.join(reconst_savepath, "{}.wav".format(savename)), sampling_rate, wavr)
            raw_text = basename.split("_")[-1]
            statement = id2text[targets[1][0].item()]
            with open(os.path.join(synth_savepath, "{}_onomatopoeia.txt".format(savename)), "w") as f:
                f.write(raw_text)
            # with open(os.path.join(synth_savepath, "{}_ambientsound.txt".format(savename)), "w") as f:
            #     f.write(statement)
            with open(os.path.join(reconst_savepath, "{}_onomatopoeia.txt".format(savename)), "w") as f:
                f.write(raw_text)
            # with open(os.path.join(reconst_savepath, "{}_ambientsound.txt".format(savename)), "w") as f:
            #     f.write(statement)

    else:
        sampling_rate = preprocess_config["preprocessing"]["audio"]["sampling_rate"]
        for wavp, basename in zip(wav_predictions, basenames):
            savename = "_".join(basename.split("_")[:-1])
            wavfile.write(os.path.join(synth_savepath, "{}.wav".format(savename)), sampling_rate, wavp)
            raw_text = basename.split("_")[-1]
            statement = id2text[targets[1][0].item()]
            with open(os.path.join(synth_savepath, "{}_onomatopoeia.txt".format(savename)), "w") as f:
                f.write(raw_text)
            # with open(os.path.join(synth_savepath, "{}_ambientsound.txt".format(savename)), "w") as f:
            #     f.write(statement)

def synth_for_eval_strech(targets, predictions, vocoder, model_config, preprocess_config, synth_savepath):
    basenames = targets[0]

    from .model import vocoder_infer

    mel_predictions = predictions[1].transpose(1, 2)
    lengths = predictions[8] * preprocess_config["preprocessing"]["stft"]["hop_length"]
    wav_predictions = vocoder_infer(
        mel_predictions, vocoder, model_config, preprocess_config, lengths=lengths
    )
    STFT = Audio.stft.TacotronSTFT(
      preprocess_config["preprocessing"]["stft"]["filter_length"],
      preprocess_config["preprocessing"]["stft"]["hop_length"],
      preprocess_config["preprocessing"]["stft"]["win_length"],
      preprocess_config["preprocessing"]["mel"]["n_mel_channels"],
      preprocess_config["preprocessing"]["audio"]["sampling_rate"],
      preprocess_config["preprocessing"]["mel"]["mel_fmin"],
      preprocess_config["preprocessing"]["mel"]["mel_fmax"],
    )
    sampling_rate = preprocess_config["preprocessing"]["audio"]["sampling_rate"]
    wav_p_len = []
    for wavp, basename in zip(wav_predictions, basenames):
        savename = "_".join(basename.split("_")[:-1])
        wavfile.write(os.path.join(synth_savepath, "{}.wav".format(savename)), sampling_rate, wavp)
        wav_p_len.append( (int(basename.split("_")[-2]), len(wavp)/sampling_rate) )
        raw_text = basename.split("_")[-1]
        with open(os.path.join(synth_savepath, "{}_onomatopoeia.txt".format(savename)), "w") as f:
            f.write(raw_text)
        # with open(os.path.join(synth_savepath, "{}_ambientsound.txt".format(savename)), "w") as f:
        #     f.write(statement)
    return wav_p_len

def synth_for_eval_continue(targets, predictions, vocoder, model_config, preprocess_config, synth_savepath):
    basenames = targets[0]

    from .model import vocoder_infer

    mel_predictions = predictions[1].transpose(1, 2)
    lengths = predictions[8] * preprocess_config["preprocessing"]["stft"]["hop_length"]
    wav_predictions = vocoder_infer(
        mel_predictions, vocoder, model_config, preprocess_config, lengths=lengths
    )
    STFT = Audio.stft.TacotronSTFT(
      preprocess_config["preprocessing"]["stft"]["filter_length"],
      preprocess_config["preprocessing"]["stft"]["hop_length"],
      preprocess_config["preprocessing"]["stft"]["win_length"],
      preprocess_config["preprocessing"]["mel"]["n_mel_channels"],
      preprocess_config["preprocessing"]["audio"]["sampling_rate"],
      preprocess_config["preprocessing"]["mel"]["mel_fmin"],
      preprocess_config["preprocessing"]["mel"]["mel_fmax"],
    )
    sampling_rate = preprocess_config["preprocessing"]["audio"]["sampling_rate"]
    wav_p_len = []
    for wavp, basename in zip(wav_predictions, basenames):
        savename = "_".join(basename.split("_")[:-1])
        wavfile.write(os.path.join(synth_savepath, "{}.wav".format(savename)), sampling_rate, wavp)
        raw_text = basename.split("_")[-1]
        wav_p_len.append( (len(raw_text), len(wavp)/sampling_rate) )  
        with open(os.path.join(synth_savepath, "{}_onomatopoeia.txt".format(savename)), "w") as f:
            f.write(raw_text)
        # with open(os.path.join(synth_savepath, "{}_ambientsound.txt".format(savename)), "w") as f:
        #     f.write(statement)
    return wav_p_len

def plot_mel_withinput(input_img, data, stats, titles):
    # plot visual-text
    ratio = input_img.shape[1]/input_img.shape[0]
    fig_width = 1.2*ratio if ratio > 3 else 1.2*3
    fig, ax = plt.subplots(1,1, figsize=(fig_width,1.5))
    ax.imshow(np.squeeze(input_img), cmap="gray")
    ax.set_title(titles[0], fontsize="medium")
    ax.tick_params(labelsize="x-small", left=False, labelleft=False)
    buf = io.BytesIO()
    plt.tight_layout()
    plt.savefig(buf, format='png')
    plt.close()
    enc = np.frombuffer(buf.getvalue(), dtype=np.uint8) # bufferからの読み出し
    im_visualtext = cv2.imdecode(enc, 1) # デコード

    fig, axes = plt.subplots(1, len(data), squeeze=False, figsize=(fig_width, 3.5))
    if titles is None:
        titles = [None for i in range(len(data))]
    energy_min, energy_max = stats

    def add_axis(fig, old_ax):
        ax = fig.add_axes(old_ax.get_position(), anchor="W")
        ax.set_facecolor("None")
        return ax

    for i in range(0,len(data)):
        mel, energy, energy_break = data[i]
        fig_aspect = fig_width/(len(data)*3.5)
        im_aspect = mel.shape[1]/(fig_aspect*mel.shape[0])
        axes[0][i].imshow(mel, origin="lower", aspect=im_aspect)
        axes[0][i].set_ylim(0, mel.shape[0]-1)
        axes[0][i].set_xlim(0, mel.shape[1]-1)
        axes[0][i].set_title(titles[i+1], fontsize="medium")
        axes[0][i].tick_params(labelsize="x-small", left=False, labelleft=False)
        axes[0][i].set_anchor("W")

        ax2 = add_axis(fig, axes[0][i])
        ax2.plot(energy, color="violet")
        ax2.set_xlim(0, mel.shape[1]-1)
        ax2.set_ylim(energy_min, energy_max)
        ax2.set_ylabel("Energy", color="darkviolet")
        ax2.yaxis.set_label_position("right")
        ax2.tick_params(
            labelsize="x-small",
            colors="darkviolet",
            bottom=False,
            labelbottom=False,
            left=False,
            labelleft=False,
            right=True,
            labelright=True,
        )
        for bre_point in energy_break:
            ax2.axvline(x=bre_point, color="violet", alpha=0.5, linestyle=":")
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    plt.close()
    enc = np.frombuffer(buf.getvalue(), dtype=np.uint8) # bufferからの読み出し
    im_mel = cv2.imdecode(enc, 1) # デコード
    im = cv2.vconcat([im_visualtext, im_mel])
    return im

def plot_mel(data, stats, titles):
    fig, axes = plt.subplots(1, len(data), squeeze=False)
    if titles is None:
        titles = [None for i in range(len(data))]
    energy_min, energy_max = stats

    def add_axis(fig, old_ax):
        ax = fig.add_axes(old_ax.get_position(), anchor="W")
        ax.set_facecolor("None")
        return ax

    for i in range(len(data)):
        mel, energy, energy_break = data[i]
        axes[0][i].imshow(mel, origin="lower")
        axes[0][i].set_aspect(2.5, adjustable="box")
        axes[0][i].set_ylim(0, mel.shape[0])
        axes[0][i].set_title(titles[i], fontsize="medium")
        axes[0][i].tick_params(labelsize="x-small", left=False, labelleft=False)
        axes[0][i].set_anchor("W")

        ax2 = add_axis(fig, axes[0][i])
        ax2.plot(energy, color="darkviolet")
        ax2.set_xlim(0, mel.shape[1])
        ax2.set_ylim(energy_min, energy_max)
        ax2.set_ylabel("Energy", color="darkviolet")
        ax2.yaxis.set_label_position("right")
        ax2.tick_params(
            labelsize="x-small",
            colors="darkviolet",
            bottom=False,
            labelbottom=False,
            left=False,
            labelleft=False,
            right=True,
            labelright=True,
        )
        for bre_point in energy_break:
            ax2.axvline(x=bre_point, color="violet", alpha=0.5, linestyle=":")
    fig.canvas.draw()
    im_mel = save_figure_to_numpy(fig)
    plt.close()
    return im_mel


def pad_1D(inputs, PAD=0):
    def pad_data(x, length, PAD):
        x_padded = np.pad(
            x, (0, length - x.shape[0]), mode="constant", constant_values=PAD
        )
        return x_padded

    max_len = max((len(x) for x in inputs))
    padded = np.stack([pad_data(x, max_len, PAD) for x in inputs])

    return padded

def pad_2D(inputs, maxlen=None):
    def pad(x, max_len):
        PAD = 0
        if np.shape(x)[0] > max_len:
            raise ValueError("not max_len")

        s = np.shape(x)[1]
        x_padded = np.pad(
            x, (0, max_len - np.shape(x)[0]), mode="constant", constant_values=PAD
        )
        return x_padded[:, :s]

    if maxlen:
        output = np.stack([pad(x, maxlen) for x in inputs])
    else:
        max_len = max(np.shape(x)[0] for x in inputs)
        output = np.stack([pad(x, max_len) for x in inputs])

    return output

def pad_2D_gray_image(inputs):
    def pad(x, max_len):
        PAD = 0
        s = np.shape(x)[0]
        x_padded = np.pad(
            x, [(0,0),(0, max_len - np.shape(x)[1])], mode="constant", constant_values=PAD
        )
        return x_padded[:s,:]
    
    max_len = max(np.shape(x)[1] for x in inputs)
    output = np.stack([pad(x, max_len) for x in inputs])

    return output

def pad_2D_image(inputs):
    # with open("check/inimg.pkl", "wb") as f:
    #     pickle.dump(inputs, f)
    def pad(x, max_len, max_height):
        PAD = 0
        x_padded = np.pad(
            x, [(0,max_height-np.shape(x)[0]),(0, max_len - np.shape(x)[1])], mode="constant", constant_values=PAD
        )
        return x_padded[:,:]
    #print(f'\ninputs[0].shape:{inputs[0].shape}')
    if len(inputs[0].shape) == 2: # grayscale-img
        max_len = max(np.shape(x)[1] for x in inputs)
        max_height = max(np.shape(x)[0] for x in inputs)
        output = np.stack([pad(x, max_len, max_height) for x in inputs])
    elif len(inputs[0].shape) == 3: # color-img
        output = np.array([])
        for i in range(inputs[0].shape[2]):
            max_len = max(np.shape(x)[1] for x in inputs)
            max_height = max(np.shape(x)[0] for x in inputs)
            out_tmp = np.stack([pad(x[:,:,i], max_len, max_height) for x in inputs])
            if i==0:
                output = np.expand_dims(out_tmp,3)
            else:
                out_tmp = np.expand_dims(out_tmp,3)
                output = np.concatenate([output, out_tmp], axis=3)
    if max_len != output.shape[2]:
        print()
    # with open("check/outimg.pkl","wb") as f:
    #     pickle.dump(output, f)
    # print("debug")
    return output

    
def pad(input_ele, mel_max_length=None):
    if mel_max_length:
        max_len = mel_max_length
    else:
        max_len = max([input_ele[i].size(0) for i in range(len(input_ele))])

    out_list = list()
    for i, batch in enumerate(input_ele):
        if len(batch.shape) == 1:
            one_batch_padded = F.pad(
                batch, (0, max_len - batch.size(0)), "constant", 0.0
            )
        elif len(batch.shape) == 2:
            one_batch_padded = F.pad(
                batch, (0, 0, 0, max_len - batch.size(0)), "constant", 0.0
            )
        out_list.append(one_batch_padded)
    out_padded = torch.stack(out_list)
    return out_padded

def save_figure_to_numpy(fig):
    # save it to a numpy array.
    data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    return data

def plot_alignment_to_numpy(alignment, info=None):
    fig, ax = plt.subplots(figsize=(6, 4))
    im = ax.imshow(alignment, aspect='auto', origin='lower',
                   interpolation='none')
    fig.colorbar(im, ax=ax)
    xlabel = 'Decoder timestep'
    if info is not None:
        xlabel += '\n\n' + info
    plt.xlabel(xlabel)
    plt.ylabel('Encoder timestep')
    plt.tight_layout()

    fig.canvas.draw()
    data = save_figure_to_numpy(fig)
    plt.close()
    return data
