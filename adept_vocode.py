"""
Re-vocode all Daft-Exprt synthesized outputs using HiFi-GAN (LJ_FT_T2_V1).

Reads the predicted mel-spec .npz files saved alongside the Griffin-Lim wavs,
runs HiFi-GAN inference, and writes new .wav files with the suffix _hifigan.wav.

Usage:
  python3 adept_vocode.py
"""

import sys
import os
import glob
import json
import numpy as np
import torch
from scipy.io.wavfile import write as wav_write

# HiFi-GAN source
HIFIGAN_DIR  = '/root/hifi-gan'
CHECKPOINT   = '/root/hifi-gan-checkpoint/LJ_FT_T2_V1/generator_v1'
CONFIG       = '/root/hifi-gan-checkpoint/LJ_FT_T2_V1/config.json'
OUTPUT_ROOT  = '/root/adept_output'

sys.path.insert(0, HIFIGAN_DIR)
from models import Generator
from env import AttrDict

MAX_WAV_VALUE = 32768.0


def load_hifigan(checkpoint_path, config_path, device):
    with open(config_path) as f:
        h = AttrDict(json.loads(f.read()))
    torch.manual_seed(h.seed)
    generator = Generator(h).to(device)
    state_dict = torch.load(checkpoint_path, map_location=device)
    generator.load_state_dict(state_dict['generator'])
    generator.eval()
    generator.remove_weight_norm()
    print(f"HiFi-GAN loaded from {checkpoint_path}")
    return generator, h


def vocode_all(generator, h, device):
    # find every predicted mel-spec npz (excludes reference npz files which
    # live in _ref_npz/ and _smoke_test/)
    npz_files = glob.glob(os.path.join(OUTPUT_ROOT, '*', '*', '*_spk_*_ref_*.npz'))
    npz_files.sort()
    print(f"Found {len(npz_files)} mel-spec files to vocode.")

    with torch.no_grad():
        for idx, npz_path in enumerate(npz_files):
            out_wav = npz_path.replace('.npz', '_hifigan.wav')
            if os.path.isfile(out_wav):
                print(f"[{idx+1}/{len(npz_files)}] skip: {os.path.basename(out_wav)}")
                continue

            data = np.load(npz_path)
            mel = data['mel_spec']          # (n_mels, T)
            x = torch.FloatTensor(mel).unsqueeze(0).to(device)  # (1, n_mels, T)

            y = generator(x).squeeze()      # (T_wav,)
            audio = (y * MAX_WAV_VALUE).cpu().numpy().astype('int16')

            wav_write(out_wav, h.sampling_rate, audio)
            print(f"[{idx+1}/{len(npz_files)}] {os.path.relpath(out_wav, OUTPUT_ROOT)}")

    print("\nDone.")


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    generator, h = load_hifigan(CHECKPOINT, CONFIG, device)
    vocode_all(generator, h, device)


if __name__ == '__main__':
    main()
