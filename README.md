# MultiSpeaker Tacotron2 in Persian Language
This repository implements [Transfer Learning from Speaker Verification to
Multispeaker Text-To-Speech Synthesis](https://arxiv.org/pdf/1806.04558.pdf) (SV2TTS) for the Persian language. The core codebase is derived from [this repository](https://github.com/Adibian/Persian-MultiSpeaker-Tacotron2), which has been updated to address deprecated features and complete setup for Persian language compatibility. The original codebase, sourced from [this repository](https://github.com/CorentinJ/Real-Time-Voice-Cloning/tree/master), has been modified to support Persian language requirements.

<img src="https://github.com/majidAdibian77/persian-SV2TTS/blob/master/results/model.JPG" width="800"> 

---

## Training
**1. Character-set definition:**

Open the `synthesizer/persian_utils/symbols.py` file and update the `_characters` variable to include all the characters that exist in your text files. Most of Persian characters and symbols are already included in this variable as follows:
```
_characters = "ءابتثجحخدذرزسشصضطظعغفقلمنهويِپچژکگیآۀأؤإئًَُّ!(),-.:;?  ̠،…؛؟‌٪#ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz_–@+/\u200c"
```

**2. Data structures:**
```
dataset/persian_date/
    train_data/
        speaker1/book-1/
            sample1.txt
            sample1.wav
            ...
        ...
    test_data/
        ...
```

**3. Preprocessing:**
```
python3 synthesizer_preprocess_audio.py dataset --datasets_name persian_data --subfolders train_data --no_alignments --skip_existing --n_processes 4 --out_dir dataset/train/SV2TTS/synthesizer
python3 synthesizer_preprocess_audio.py dataset --datasets_name persian_data --subfolders test_data --no_alignments --skip_existing --n_processes 4 --out_dir dataset/test/SV2TTS/synthesizer
```
2. **Embedding Preprocessing**  
```
python3 synthesizer_preprocess_embeds.py dataset/train/SV2TTS/synthesizer
python3 synthesizer_preprocess_embeds.py dataset/test/SV2TTS/synthesizer
```

**4. Train synthesizer:**
```
python3 synthesizer_train.py my_run dataset/train/SV2TTS/synthesizer
```

## Inference

To generate a wav file, place all trained models in the `saved_models/final_models` directory. If you haven’t trained the speaker encoder or vocoder models, you can use pretrained models from `saved_models/default`. These models include `encoder.pt`, your latest synthesizer checkpoint like `synthesizer_000300.pt`, and a vocoder as follows.

### Using WavRNN as Vocoder

```
python3 inference.py --vocoder "WavRNN" --text "یک نمونه از خروجی" --ref_wav_path "/path/to/sample/reference.wav" --test_name "test1"
```

### Using HiFiGAN as Vocoder (Recommended)
WavRNN is an old vocoder and if you want to use HiFiGAN you must first download a pretrained model in English.
1. **Install Parallel WaveGAN**  
```
pip install parallel_wavegan
```
2. **Download Pretrained HiFiGAN Model**  
```
from parallel_wavegan.utils import download_pretrained_model
download_pretrained_model("vctk_hifigan.v1", "saved_models/final_models/vocoder_HiFiGAN")
```
3. **Run Inference with HiFiGAN**
```
python3 inference.py --vocoder "HiFiGAN" --text "یک نمونه از خروجی" --ref_wav_path "/path/to/sample/reference.wav" --test_name "test1"
```

## ManaTTS-Trained Model

This architecture has been used to train a Persian Text-to-Speech (TTS) model on the [**ManaTTS dataset**](https://huggingface.co/datasets/MahtaFetrat/Mana-TTS), the largest publicly available single-speaker Persian corpus. The trained model weights and detailed inference instructions can be found in the following repositories:

- [Hugging Face Repository](https://huggingface.co/MahtaFetrat/Persian-Tacotron2-on-ManaTTS)
- [GitHub Repository](https://github.com/MahtaFetrat/ManaTTS-Persian-Tacotron2-Model)

## References:
- [Transfer Learning from Speaker Verification to Multispeaker Text-To-Speech Synthesis](https://arxiv.org/pdf/1806.04558.pdf) Ye Jia, *et al*.,
- [Real-Time-Voice-Cloning repository](https://github.com/CorentinJ/Real-Time-Voice-Cloning/tree/master),
- [ParallelWaveGAN repository](https://github.com/kan-bayashi/ParallelWaveGAN)
- [Persian-MultiSpeaker-Tacotron2](https://github.com/Adibian/Persian-MultiSpeaker-Tacotron2)

## License  
This project is based on [Real-Time-Voice-Cloning](https://github.com/CorentinJ/Real-Time-Voice-Cloning),  
which is licensed under the MIT License.  
```
Modified & original work Copyright (c) 2019 Corentin Jemine (https://github.com/CorentinJ)  
Original work Copyright (c) 2018 Rayhane Mama (https://github.com/Rayhane-mamah)  
Original work Copyright (c) 2019 fatchord (https://github.com/fatchord)  
Original work Copyright (c) 2015 braindead (https://github.com/braindead)  
Modified work Copyright (c) 2025 Majid Adibian (https://github.com/Adibian)  
Modified work Copyright (c) 2025 Mahta Fetrat (https://github.com/MahtaFetrat)
``` 

  
