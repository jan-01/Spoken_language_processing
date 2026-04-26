"""
ADEPT prosody transfer synthesis pipeline for Daft-Exprt.

For each (category, condition, sentence_item):
  - reference = ADEPT wav for that condition
  - text      = the sentence from the neutral condition (fixed target text)
  - output    = synthesized wav saved under adept_output/{category}/{condition}/

Usage:
  QT_QPA_PLATFORM=offscreen python3 adept_synthesis.py
"""

import os
import sys
import re
import torch

PROJECT_ROOT = '/root/ubisoft-laforge-daft-exprt'
sys.path.append(os.path.join(PROJECT_ROOT, 'src'))

from daft_exprt.generate import extract_reference_parameters, generate_mel_specs
from daft_exprt.hparams import HyperParams
from daft_exprt.model import DaftExprt
from daft_exprt.utils import get_nb_jobs

# ── paths ────────────────────────────────────────────────────────────────────
CHECKPOINT  = '/root/DaftExprt_LJ_ESD_22kHz'
ADEPT_ROOT  = '/root/ADEPT'
OUTPUT_ROOT = '/root/adept_output'
MFA_DICT    = '/root/Documents/MFA/pretrained_models/dictionary/english.dict'

# speaker 0 is used for all outputs so prosody differences come only from the
# reference, not from speaker identity
SPEAKER_ID  = 0

# categories and their prosody conditions
CATEGORIES = {
    'emotion':               ['anger', 'fear', 'joy', 'neutral', 'sadness'],
    'topical_emphasis':      ['beginning', 'middle', 'end'],
    'propositional_attitude': ['incredulity', 'neutral', 'sarcasm', 'surprise'],
}

# ── G2P via MFA dictionary ────────────────────────────────────────────────────

# words missing from the MFA dict but present in ADEPT sentences
MANUAL_PHONES = {
    'HANUKKAH':   'HH AA1 N AH0 K AH0',
    'CHARDONNAY': 'SH AA0 R D AH0 N EY1',
    'LARRY':      'L AE1 R IY0',
    'DEVANG':     'D AH0 V AE1 NG',
    'DUNKED':     'D AH1 NG K T',
    'JESSE':      'JH EH1 S IY0',
    'HANNAH':     'HH AE1 N AH0',
    'TONY':       'T OW1 N IY0',
    'GALLERIES':  'G AE1 L ER0 IY0 Z',
}


def load_mfa_dict(path):
    d = {}
    with open(path, encoding='utf-8') as f:
        for line in f:
            parts = line.rstrip('\n').split(None, 1)
            if len(parts) == 2:
                word, phones = parts
                if word not in d:  # keep first (most common) pronunciation
                    d[word] = phones
    d.update(MANUAL_PHONES)
    return d


def word_to_phones(word, mfa_dict):
    """Return ARPAbet phoneme list for a word, or None if not found."""
    key = word.upper()
    if key in mfa_dict:
        return mfa_dict[key].split()
    # try stripping trailing possessive / plural markers as a fallback
    for suffix in ("'S", "S", "ED", "ING"):
        if key.endswith(suffix) and key[:-len(suffix)] in mfa_dict:
            return mfa_dict[key[:-len(suffix)]].split()
    return None


def text_to_sentence(text, mfa_dict):
    """
    Convert a plain-text sentence into the Daft-Exprt sentence list format:
        [['P', 'H', 'O', 'N', 'E', 'M', 'E', 'S'], ' ', ['N', 'E', 'X', 'T'], '.', '~']

    Punctuation tokens (., ?, !, ,) are kept as plain strings.
    The end-of-sentence marker '~' is appended.
    Returns (sentence_list, unknown_words).
    """
    text = text.strip()

    # separate leading/trailing sentence-ending punctuation
    end_punct = ''
    if text and text[-1] in '.?!':
        end_punct = text[-1]
        text = text[:-1].strip()

    # tokenise: split on spaces, keeping inline commas attached to the word
    raw_tokens = text.split()

    sentence = []
    unknown = []
    for raw in raw_tokens:
        # split inline comma (e.g. "friends," → "friends" + ",")
        inline_punct = ''
        if raw and raw[-1] in ',.;:':
            inline_punct = raw[-1]
            raw = raw[:-1]

        phones = word_to_phones(raw, mfa_dict)
        if phones is None:
            unknown.append(raw)
            # fall back: treat as a space-separated token so synthesis doesn't crash
            phones = ['AH0']  # schwa placeholder

        sentence.append(phones)
        sentence.append(' ')   # word boundary

        if inline_punct:
            sentence.append(inline_punct)

    # remove trailing space boundary if present
    if sentence and sentence[-1] == ' ':
        sentence.pop()

    if end_punct:
        sentence.append(end_punct)
    sentence.append('~')

    return sentence, unknown


# ── dataset helpers ───────────────────────────────────────────────────────────

def get_neutral_text(category, item_id):
    """
    Return the canonical (neutral) text for a sentence item.
    For topical_emphasis the text is identical across conditions; for
    emotion/propositional_attitude 'neutral' holds the plain version.
    """
    neutral_dir = os.path.join(ADEPT_ROOT, 'txt', category, 'neutral')
    if not os.path.isdir(neutral_dir):
        # topical_emphasis has no 'neutral' — use first available condition
        neutral_dir = os.path.join(ADEPT_ROOT, 'txt', category,
                                   os.listdir(os.path.join(ADEPT_ROOT, 'txt', category))[0])
    txt_path = os.path.join(neutral_dir, f'{item_id}.txt')
    if not os.path.isfile(txt_path):
        return None
    with open(txt_path) as f:
        return f.read().strip()


def iter_adept_jobs(categories):
    """
    Yield (category, condition, item_id, ref_wav, target_text) for every
    valid combination where both the wav and a target text exist.
    """
    for category, conditions in categories.items():
        wav_cat = os.path.join(ADEPT_ROOT, 'wav_44khz', category)
        for condition in conditions:
            cond_dir = os.path.join(wav_cat, condition)
            if not os.path.isdir(cond_dir):
                continue
            for wav_file in sorted(os.listdir(cond_dir)):
                if not wav_file.endswith('.wav'):
                    continue
                item_id = wav_file.replace('.wav', '')
                ref_wav = os.path.join(cond_dir, wav_file)
                target_text = get_neutral_text(category, item_id)
                if target_text is None:
                    continue
                yield category, condition, item_id, ref_wav, target_text


# ── model loading ─────────────────────────────────────────────────────────────

def load_model():
    print("Loading model...")
    checkpoint_dict = torch.load(CHECKPOINT, map_location='cuda:0')
    config = checkpoint_dict['config_params']
    hparams = HyperParams.__new__(HyperParams)
    hparams.__dict__.update(config)
    hparams.mfa_dictionary    = '/root/Documents/MFA/pretrained_models/dictionary/english.dict'
    hparams.mfa_g2p_model     = '/root/Documents/MFA/pretrained_models/g2p/english_g2p.zip'
    hparams.mfa_acoustic_model = '/root/Documents/MFA/pretrained_models/acoustic/english.zip'
    torch.cuda.set_device(0)
    model = DaftExprt(hparams).cuda(0)
    state_dict = {k.replace('module.', ''): v
                  for k, v in checkpoint_dict['state_dict'].items()}
    model.load_state_dict(state_dict)
    print("Model loaded.")
    return model, hparams


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    mfa_dict = load_mfa_dict(MFA_DICT)
    model, hparams = load_model()
    n_jobs = get_nb_jobs('max')

    # collect all jobs, grouped so we can batch them per category/condition
    # and cache extracted reference .npz files
    jobs = list(iter_adept_jobs(CATEGORIES))
    print(f"\nTotal synthesis jobs: {len(jobs)}")

    # pre-extract all reference parameters (idempotent — skips existing .npz)
    # Use per-category/condition subdirs so same-named files across conditions
    # (e.g. emotion/anger/ad00_0000.wav vs emotion/fear/ad00_0000.wav) don't
    # overwrite each other in the cache.
    print("\nExtracting reference parameters...")
    seen_wavs = set()
    for category, condition, _, ref_wav, _ in jobs:
        if ref_wav not in seen_wavs:
            ref_cache_dir = os.path.join(OUTPUT_ROOT, '_ref_npz', category, condition)
            extract_reference_parameters(ref_wav, ref_cache_dir, hparams)
            seen_wavs.add(ref_wav)
    print(f"  {len(seen_wavs)} reference files processed.")

    # synthesize
    all_unknown = {}
    total = len(jobs)
    for idx, (category, condition, item_id, ref_wav, target_text) in enumerate(jobs):
        out_dir = os.path.join(OUTPUT_ROOT, category, condition)
        os.makedirs(out_dir, exist_ok=True)

        out_name = f"{item_id}"  # e.g. ad00_0000
        # Daft-Exprt appends speaker and ref to the filename
        ref_basename = os.path.basename(ref_wav).replace('.wav', '')
        actual_out_wav = os.path.join(out_dir,
                                      f"{out_name}_spk_{SPEAKER_ID}_ref_{ref_basename}.wav")
        if os.path.isfile(actual_out_wav):
            print(f"[{idx+1}/{total}] skip (exists): {category}/{condition}/{out_name}")
            continue

        sentence, unknown = text_to_sentence(target_text, mfa_dict)
        if unknown:
            all_unknown[f"{category}/{condition}/{item_id}"] = unknown

        ref_npz = os.path.join(OUTPUT_ROOT, '_ref_npz', category, condition,
                               os.path.basename(ref_wav).replace('.wav', '.npz'))

        print(f"[{idx+1}/{total}] {category}/{condition}/{out_name}  \"{target_text}\"")

        generate_mel_specs(
            model,
            [sentence],
            [out_name],
            [SPEAKER_ID],
            [ref_npz],
            out_dir,
            hparams,
            None, None, None,
            50, n_jobs, False, False  # use_griffin_lim=False, HiFi-GAN used separately
        )

    if all_unknown:
        print("\nWARNING — unknown words (replaced with schwa placeholder):")
        for key, words in all_unknown.items():
            print(f"  {key}: {words}")

    print(f"\nDone. Outputs in {OUTPUT_ROOT}/")


if __name__ == '__main__':
    main()
