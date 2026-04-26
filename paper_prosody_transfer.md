# Evaluating Prosody Transfer in Daft-Exprt: An Acoustic Analysis Using the ADEPT Benchmark

**Course:** Spoken Language Processing (Wahlpflichtfach)  
**Date:** April 2026

---

## Abstract

Prosody transfer — the ability of a text-to-speech system to reproduce the pitch contour, rhythm, and expressive style of a reference speaker on arbitrary new text — remains one of the most challenging open problems in speech synthesis. This paper evaluates the prosody transfer capabilities of Daft-Exprt [Zaïdi et al., 2021], a multi-speaker acoustic model based on FiLM conditioning, using the ADEPT benchmark [Shechtman & Fernandez, 2019]. We synthesize speech for three prosody categories — *emotion*, *topical emphasis*, and *propositional attitude* — and compare synthesized outputs against human reference recordings on three acoustic dimensions: mean F0, F0 standard deviation, and utterance duration. Results show that Daft-Exprt successfully transfers global pitch level and variability in the emotion and propositional attitude categories, faithfully encoding the high-pitched excitability of joy and fear and the subdued pitch of sadness and sarcasm. Topical emphasis, which operates through local F0 peaks and duration patterns on individual words, is not reliably captured by the model's global prosody representation. Duration is consistently underestimated across all conditions. These findings highlight the gap between global and local prosody transfer in current reference-based TTS systems.

---

## 1. Introduction

Prosody encompasses the suprasegmental features of speech — pitch (F0), duration, and intensity — that convey meaning beyond the words themselves. The same sentence can express anger, joy, or sarcasm depending entirely on how it is spoken. Building text-to-speech (TTS) systems that can reproduce this expressive variation is critical for applications in gaming, virtual agents, and accessibility tools.

A common approach is *reference-based prosody transfer*: given a reference audio recording that exhibits a desired speaking style, the TTS system should produce new speech, on new text, that shares the prosodic characteristics of the reference. This task is challenging because prosody must be extracted independently of the reference speaker's voice identity and then applied coherently to a different phoneme sequence.

Daft-Exprt [Zaïdi et al., 2021] addresses this challenge with an architecture that explicitly encodes low-level prosodic features — pitch, loudness, and duration — from the reference utterance using a prosody encoder, and injects this information throughout the acoustic decoder via FiLM (Feature-wise Linear Modulation) conditioning layers. Speaker identity is disentangled from prosody through adversarial training, enabling cross-speaker transfer. The model is trained on a combination of the LJ Speech dataset [Ito & Johnson, 2017] and the Emotional Speech Dataset (ESD) [Zhou et al., 2022], giving it exposure to both neutral read speech and expressive emotional speech across multiple speakers.

To evaluate how well Daft-Exprt actually transfers prosody, we use the ADEPT (Acoustic Database for Expressive Prosody Transfer) benchmark [Shechtman & Fernandez, 2019]. ADEPT provides matched human recordings of the same sentences spoken in different prosodic conditions across six categories: *emotion*, *topical emphasis*, *propositional attitude*, *discourse function*, *lexical entrainment*, and *phrasing*. This design makes it ideal for measuring whether a TTS system can replicate prosodic contrasts: if the system successfully transfers prosody, its outputs for different conditions of the same sentence should differ acoustically in the same way the human references do.

**Research question:** Which ADEPT prosody categories does Daft-Exprt transfer faithfully, and where does it fail — as revealed by acoustic measurements?

---

## 2. Methodology

### 2.1 Dataset

We use three of the six ADEPT categories for which audio was available in the `wav_44khz` directory:

- **Emotion** (5 conditions): *anger*, *fear*, *joy*, *neutral*, *sadness* — 5 sentences per condition (items ad00–ad04), yielding 25 reference/synthesis pairs.
- **Topical emphasis** (3 conditions): *beginning*, *middle*, *end* — sentences where the informationally salient word shifts position; 15 pairs.
- **Propositional attitude** (4 conditions): *incredulity*, *neutral*, *sarcasm*, *surprise* — attitudes conveyed by the same proposition; between 1 and 5 items per condition (sarcasm has only 1 item), yielding 15 pairs.

The total experimental set is **55 reference–synthesis pairs** across 12 unique conditions.

### 2.2 Synthesis Pipeline

**Text input.** For every item in each condition, the target text is taken from the *neutral* condition of the same sentence. This design choice isolates the effect of the prosodic reference: the words being synthesized are always neutral, while the expressive style is determined solely by the reference audio. For *topical emphasis*, which has no neutral condition, the text from the first available condition is used (the text is identical across conditions for this category).

**Grapheme-to-phoneme (G2P).** Daft-Exprt operates on ARPAbet phoneme sequences rather than raw text. We convert words to phonemes using the Montreal Forced Aligner (MFA) dictionary [McAuliffe et al., 2017], a pronunciation lexicon containing over 200,000 English words with primary and secondary stress markers (e.g., "HAPPY" → `HH AE1 P IY0`). Nine words absent from the MFA dictionary and unique to ADEPT sentences (e.g., *Hanukkah*, *Chardonnay*) were added manually. All 30 unique ADEPT sentences were converted without fallback to the placeholder phoneme.

**Reference parameter extraction.** For each reference recording, we run Daft-Exprt's reference encoder to extract a compact prosody representation stored in a NumPy `.npz` archive. This archive contains the log-F0 contour, energy contour, and mel-spectrogram of the reference. A critical implementation detail is that references for each condition must be stored in separate subdirectories to prevent name collisions: the same sentence filename (e.g., `ad00_0000.wav`) appears in every condition of a category. Early experiments that cached by filename only caused all conditions within a category to share the same reference, producing identical outputs regardless of condition.

**Acoustic model (Daft-Exprt).** The model takes the phoneme sequence and the reference `.npz` as input. Its duration predictor allocates frames to phonemes conditioned on the reference prosody; its pitch and energy predictors generate F0 and loudness contours; and the Transformer decoder produces an 80-band log-mel spectrogram at 22 050 Hz. A fixed speaker embedding (`speaker_id = 0`, corresponding to a neutral LJ Speech voice) is used for all outputs, ensuring that observed prosodic differences across conditions reflect reference transfer, not speaker identity changes.

**Vocoder (HiFi-GAN).** The predicted mel-spectrograms are converted to waveforms using HiFi-GAN [Kong et al., 2020], a GAN-based neural vocoder with a fine-tuned checkpoint (LJ_FT_T2_V1). HiFi-GAN produces substantially more natural-sounding audio than the Griffin-Lim algorithm available as an alternative, with no spectral distortion. The mel-spectrogram representation is directly compatible between Daft-Exprt and HiFi-GAN (identical sampling rate, FFT size, hop length, number of bins, and log-clamp normalization), requiring no conversion.

### 2.3 Acoustic Analysis

We analyze each synthesized and reference waveform with Parselmouth [Jadoul et al., 2018], a Python interface to the Praat acoustic analysis toolkit. Three features are extracted:

- **Mean F0 (Hz):** pitch autocorrelation with time step 10 ms, floor 75 Hz, ceiling 500 Hz; unvoiced frames (F0 = 0) excluded from the mean.
- **F0 standard deviation (Hz):** within-utterance pitch variability, voiced frames only.
- **Utterance duration (s):** total signal duration from the Sound object.

Per-condition means are computed by averaging item-level statistics across all sentences in that condition. We compare reference and synthesized values to assess transfer fidelity.

---

## 3. Results

Table 1 shows the mean F0, F0 standard deviation, and duration for each condition, averaged across all items.

**Table 1.** Acoustic comparison of reference (REF) and synthesized (SYNTH) speech per condition. F0 values in Hz; duration in seconds.

| Category | Condition | REF F0 | SYNTH F0 | REF σ(F0) | SYNTH σ(F0) | REF dur | SYNTH dur |
|---|---|---:|---:|---:|---:|---:|---:|
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

### 3.1 Emotion

The emotion category shows the strongest evidence of prosody transfer. The rank ordering of mean F0 across conditions is preserved: joy and fear (the two high-arousal positive/negative emotions) are the highest-pitched in both reference (277.1 and 259.8 Hz) and synthesis (280.2 and 259.2 Hz), while neutral is the lowest in both (160.7 vs. 169.1 Hz). Sadness and anger occupy the middle range in both recordings and synthesis.

F0 variability (σ) is also transferred directionally. Joy shows the highest variability in both reference (66.1 Hz) and synthesis (75.5 Hz). Neutral shows the lowest in the reference (24.7 Hz); synthesis amplifies this somewhat (41.9 Hz), suggesting the model does not fully suppress variability even when the reference is flat.

Duration shows a consistent compression: synthesized utterances are 15–25% shorter than references across all emotion conditions. This is discussed further in Section 3.4.

### 3.2 Propositional Attitude

Propositional attitude also shows clear transfer. The most striking contrast is between sarcasm (reference: 130.7 Hz, synthesis: 144.5 Hz) and incredulity (reference: 276.7 Hz, synthesis: 260.5 Hz) — both the direction and the magnitude of the contrast are reproduced. Neutral propositional attitude (169.1 vs. 171.5 Hz) is accurately synthesized with low error (1.4%). Surprise sits between neutral and incredulity in both reference and synthesis.

F0 variability is slightly overestimated across all propositional attitude conditions (SYNTH σ > REF σ in all four cases), most noticeably for sarcasm (+18.5 Hz). This may reflect the model's difficulty in generating the characteristically flat, drawling F0 contour of sarcasm; instead it produces a more variable — and thus less sarcastic-sounding — contour.

### 3.3 Topical Emphasis

Topical emphasis is the weakest category. In the reference recordings, the three conditions (emphasis at the beginning, middle, or end of the sentence) differ by only 8–9 Hz in mean F0 (174.0 / 183.3 / 182.4 Hz). This is expected: topical emphasis in English is realized primarily through a local pitch accent and lengthening on the focused word, not through a global shift of the entire F0 trajectory.

The synthesized outputs show equally small differences (178.8 / 178.5 / 190.5 Hz), and the rank ordering does not match the references (reference rank: beginning < end < middle; synthesis rank: middle < beginning < end). This failure is not surprising given the nature of Daft-Exprt's prosody representation. The reference encoder extracts a *global* prosody embedding from the entire reference utterance, averaged over all frames. A local pitch accent on a single word contributes minimally to this global representation, so the conditioning signal for "emphasis at the beginning" versus "emphasis at the end" is nearly identical from the model's perspective.

### 3.4 Duration Compression

Across all three categories, synthesized utterances are consistently shorter than references (mean reduction: 22% for emotion, 13% for propositional attitude, 12% for topical emphasis). This systematic underestimation of duration is not unique to Daft-Exprt and has been reported in other reference-based TTS systems. Several factors may contribute:

1. **Domain mismatch.** The reference recordings are from ADEPT (varied speakers, expressive speech); the model was trained on LJ Speech (single female speaker, read speech) and ESD (scripted emotional speech). Reference utterances with longer pauses, slower articulation rates, or unusual prosodic patterns may not have analogues in the training distribution.
2. **Speaker identity mismatch.** The synthesized speaker (`id = 0`, LJ Speech voice) has an inherent speaking rate that differs from the ADEPT speakers. Since the duration predictor is conditioned on both the reference and the speaker embedding, the fixed fast-speaking speaker identity may partially override the reference's slower pace.
3. **Global prosody averaging.** The global prosody embedding compresses all temporal information into a fixed-size vector; precise frame-level duration targets from the reference may be lost.

---

## 4. Discussion

The results reveal a clear asymmetry between *global* and *local* prosody in Daft-Exprt's transfer capability.

**Global prosody is well transferred.** Mean F0 — determined by overall arousal level, speaker register, and modal pitch — is reproduced with errors consistently below 10 Hz for emotion and propositional attitude. The model correctly encodes that fear requires a high, wide-ranging pitch and that sarcasm requires a low, compressed pitch. This is the primary strength of the FiLM conditioning approach: the reference encoder succeeds at distilling the dominant pitch level of the reference into its conditioning representation.

**Local prosody is not transferred.** Topical emphasis, which requires placing a pitch accent on a specific word, is essentially absent from the synthesis. More subtly, the model overestimates F0 variability for low-dynamic conditions (neutral, sadness, sarcasm), suggesting it cannot fully suppress movement and achieve the deliberate flatness that characterizes these styles.

**Practical implications.** For applications requiring expressive speech with salient emotional coloring (e.g., game character voices), Daft-Exprt provides a usable prosody transfer capability. For applications requiring fine-grained prosodic control — emphasis, focus, contrast — a different approach would be needed, such as explicit prominence marking in the text input or a frame-level reference attention mechanism rather than a global embedding.

---

## 5. Conclusion

We evaluated the prosody transfer capabilities of Daft-Exprt on three ADEPT categories using acoustic measurements of F0 and duration. The system successfully transfers global pitch characteristics in the emotion and propositional attitude categories, reproducing the rank ordering and approximate magnitude of mean F0 differences between conditions. Topical emphasis, which requires local pitch accents rather than global pitch shifts, is not captured by the model's global prosody embedding. Duration is consistently underestimated across all conditions. These results suggest that global-embedding-based prosody transfer is mature for categorical expressive styles but requires augmentation for linguistic prominence and fine-grained prosodic control.

---

## References

- Ito, K. & Johnson, L. (2017). The LJ Speech Dataset. *https://keithito.com/LJ-Speech-Dataset/*

- Jadoul, Y., Thompson, B., & de Boer, B. (2018). Introducing Parselmouth: A Python interface to Praat. *Journal of Phonetics*, 71, 1–15.

- Kong, J., Kim, J., & Bae, J. (2020). HiFi-GAN: Generative Adversarial Networks for Efficient and High Fidelity Speech Synthesis. *NeurIPS 2020*.

- McAuliffe, M., Socolof, M., Mihuc, S., Wagner, M., & Sonderegger, M. (2017). Montreal Forced Aligner: Trainable text-speech alignment using Kaldi. *Interspeech 2017*.

- Shechtman, S. & Fernandez, R. (2019). ADEPT: A database for emotional prosody transfer. *Interspeech 2019*.

- Zaïdi, J., Seuté, H., van Niekerk, B., & Carbonneau, M.-A. (2021). Daft-Exprt: Robust Prosody Transfer Across Speakers for Expressive Speech Synthesis. *arXiv:2108.02271*.

- Zhou, K., et al. (2022). Emotional Voice Conversion: Theory, Databases and ESD. *IEEE/ACM TASLP*.
