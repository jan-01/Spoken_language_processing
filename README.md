# Spoken Language Processing — Prosody Transfer Evaluation

**Course:** Spoken Language Processing  
**Author:** jan-01  
**Date:** April 2026

---

## Overview

This project evaluates the prosody transfer capabilities of **Daft-Exprt**, a reference-based text-to-speech system, using the **ADEPT benchmark**. Synthesized speech is assessed across three prosodic dimensions — emotion, propositional attitude, and topical emphasis — using acoustic analysis (F0, duration) and automatic speech recognition (WER).

---

## Repository Structure

```
.
├── ADEPT/                        # ADEPT benchmark dataset
│   ├── txt/                      # Ground truth transcriptions per condition
│   └── wav_44khz/                # Reference audio recordings
│
├── adept_output/                 # Synthesized waveforms (HiFi-GAN vocoder)
│   ├── emotion/                  # anger, fear, joy, neutral, sadness
│   ├── propositional_attitude/   # incredulity, neutral, sarcasm, surprise
│   └── topical_emphasis/         # beginning, middle, end
│
├── adept_plots/                  # Acoustic analysis visualisations
│   ├── emotion/                  # Per-item F0 overlay plots
│   ├── propositional_attitude/
│   ├── topical_emphasis/
│   └── summary.csv               # Per-item acoustic measurements
│
├── adept_synthesis.py            # Synthesises ADEPT utterances via Daft-Exprt
├── adept_vocode.py               # Applies HiFi-GAN vocoder to mel spectrograms
├── adept_analysis.py             # Extracts F0 and duration with Parselmouth/Praat
├── calculate_adept_wer.py        # Transcribes outputs with Whisper, computes WER
├── run_synthesis.py              # Entry point for running the full synthesis pipeline
│
├── adept_wer_results.txt         # Per-file WER and CER results
├── adept_synthesis.log           # Synthesis run log
└── prosody_transfer_report.pdf   # Full written report
```

---

## Pipeline

1. **Synthesis** (`adept_synthesis.py`) — generates mel spectrograms from ADEPT text using reference audio from the dataset as the prosody conditioning signal.
2. **Vocoding** (`adept_vocode.py`) — converts mel spectrograms to waveforms using the HiFi-GAN neural vocoder.
3. **Acoustic analysis** (`adept_analysis.py`) — extracts mean F0, F0 standard deviation, and duration from both reference and synthesized audio using Parselmouth (Python/Praat interface).
4. **WER evaluation** (`calculate_adept_wer.py`) — transcribes all 115 synthesized utterances with OpenAI Whisper (base) and computes Word Error Rate against ADEPT ground truth transcriptions.

---

## Key Results

### Acoustic Analysis (F0 & Duration)

| Category | Condition | REF F0 (Hz) | SYNTH F0 (Hz) | REF σ(F0) | SYNTH σ(F0) | REF dur (s) | SYNTH dur (s) |
|---|---|---|---|---|---|---|---|
| Emotion | anger | 242.0 | 232.7 | 65.2 | 53.2 | 2.29 | 1.78 |
| Emotion | fear | 259.8 | 259.2 | 45.2 | 55.8 | 1.92 | 1.61 |
| Emotion | joy | 277.1 | 280.2 | 66.1 | 75.5 | 1.86 | 1.58 |
| Emotion | neutral | 160.7 | 169.1 | 24.7 | 41.9 | 1.60 | 1.46 |
| Emotion | sadness | 189.6 | 182.5 | 33.4 | 36.0 | 2.08 | 1.66 |
| Prop. attitude | incredulity | 276.7 | 260.5 | 79.9 | 85.2 | 2.14 | 1.87 |
| Prop. attitude | neutral | 169.1 | 171.5 | 37.9 | 46.4 | 1.79 | 1.66 |
| Prop. attitude | sarcasm | 130.7 | 144.5 | 38.8 | 57.3 | 2.65 | 1.92 |
| Prop. attitude | surprise | 236.4 | 228.0 | 67.3 | 71.6 | 2.06 | 1.86 |
| Topical emph. | beginning | 174.0 | 178.8 | 41.3 | 53.9 | 2.62 | 2.38 |
| Topical emph. | middle | 183.3 | 178.5 | 42.6 | 40.4 | 2.61 | 2.29 |
| Topical emph. | end | 182.4 | 190.5 | 37.3 | 53.2 | 2.71 | 2.28 |

### Intelligibility (WER)

| Category | Condition | WER (%) |
|---|---|---|
| Emotion | anger | 8.5 |
| Emotion | fear | 9.5 |
| Emotion | joy | 7.0 |
| Emotion | neutral | 9.9 |
| Emotion | sadness | 7.0 |
| Prop. attitude | incredulity | 14.0 |
| Prop. attitude | neutral | 11.0 |
| Prop. attitude | sarcasm | 0.0† |
| Prop. attitude | surprise | 18.5 |
| Topical emph. | beginning | 19.0 |
| Topical emph. | middle | 13.9 |
| Topical emph. | end | 13.3 |
| **Overall** | | **11.4** |

†Sarcasm evaluated on speaker ad01 only (n = 5).

---

## Dependencies

- [Daft-Exprt](https://github.com/ubisoft/ubisoft-laforge-daft-exprt) — reference-based TTS model
- [HiFi-GAN](https://github.com/jik876/hifi-gan) — neural vocoder
- [Parselmouth](https://parselmouth.readthedocs.io/) — Python interface to Praat
- [OpenAI Whisper](https://github.com/openai/whisper) — ASR for WER evaluation
- [jiwer](https://github.com/jitsi/jiwer) — WER computation

---

## Report

The full written analysis is in [`prosody_transfer_report.pdf`](prosody_transfer_report.pdf), covering methodology, per-category acoustic results, intelligibility analysis, and discussion of the model's global vs. local prosody transfer capabilities.
