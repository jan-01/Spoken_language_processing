import os
import sys
import random
import torch
import re

PROJECT_ROOT = '/root/ubisoft-laforge-daft-exprt'
sys.path.append(os.path.join(PROJECT_ROOT, 'src'))

from daft_exprt.generate import extract_reference_parameters, generate_mel_specs
from daft_exprt.hparams import HyperParams
from daft_exprt.model import DaftExprt
from daft_exprt.utils import get_nb_jobs
from shutil import copyfile

random.seed(1234)

CHECKPOINT = '/root/DaftExprt_LJ_ESD_22kHz'
TEXT_FILE  = '/root/demo/sentences_to_generate.txt'
STYLE_BANK = '/root/demo'
OUTPUT_DIR = '/root/test_output'

os.makedirs(OUTPUT_DIR, exist_ok=True)

def parse_phonemized_line(line):
    ''' Parse already-phonemized lines in {P H O N E M E S} format '''
    sentence = []
    tokens = re.findall(r'\{[^}]+\}|[^\s{]+', line)
    for token in tokens:
        if token.startswith('{') and token.endswith('}'):
            phonemes = token[1:-1].strip().split()
            sentence.append(phonemes)
        else:
            sentence.append(token)
    return sentence

# load checkpoint and model
print("Loading model...")
checkpoint_dict = torch.load(CHECKPOINT, map_location='cuda:0')

# patch hparams to skip MFA file checks
config = checkpoint_dict['config_params']
hparams = HyperParams.__new__(HyperParams)
hparams.__dict__.update(config)
hparams.mfa_dictionary  = '/root/Documents/MFA/pretrained_models/dictionary/english.dict'
hparams.mfa_g2p_model   = '/root/Documents/MFA/pretrained_models/g2p/english_g2p.zip'
hparams.mfa_acoustic_model = '/root/Documents/MFA/pretrained_models/acoustic/english.zip'

torch.cuda.set_device(0)
model = DaftExprt(hparams).cuda(0)
state_dict = {k.replace('module.', ''): v for k, v in checkpoint_dict['state_dict'].items()}
model.load_state_dict(state_dict)
print("Model loaded.")

# parse sentences
with open(TEXT_FILE, 'r') as f:
    lines = [l.strip() for l in f]

sentences, file_names = [], []
for idx, line in enumerate(lines):
    parts = line.split('|', 1)
    file_names.append(parts[0])
    sentences.append(parse_phonemized_line(parts[1]))

# extract reference parameters
print("Extracting reference parameters...")
audio_refs = [os.path.join(STYLE_BANK, x) for x in os.listdir(STYLE_BANK) if x.endswith('.wav')]
for audio_ref in audio_refs:
    extract_reference_parameters(audio_ref, STYLE_BANK, hparams)

refs = [os.path.join(STYLE_BANK, x) for x in os.listdir(STYLE_BANK) if x.endswith('.npz')]
refs = [random.choice(refs) for _ in range(len(sentences))]
speaker_ids = [random.choice(hparams.speakers_id) for _ in range(len(sentences))]

# generate
print("Synthesizing...")
n_jobs = get_nb_jobs('max')
generate_mel_specs(model, sentences, file_names, speaker_ids, refs,
                   OUTPUT_DIR, hparams, None, None, None,
                   50, n_jobs, True, False)

print(f"Done. Output in {OUTPUT_DIR}")
