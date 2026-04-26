"""
Acoustic analysis of ADEPT prosody transfer results.

For each (category, condition, item), compares:
  - reference wav  (human recording from ADEPT)
  - synthesized wav (Daft-Exprt output)

Extracts F0, duration, intensity via Parselmouth.
Generates per-category overlay plots (reference vs synthesised, all conditions).

Usage:
  python3 adept_analysis.py
"""

import os
import re
import glob
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import parselmouth
from parselmouth.praat import call

ADEPT_ROOT  = '/root/ADEPT'
OUTPUT_ROOT = '/root/adept_output'
PLOTS_DIR   = '/root/adept_plots'
SPEAKER_ID  = 0

CATEGORIES = {
    'emotion':               ['anger', 'fear', 'joy', 'neutral', 'sadness'],
    'topical_emphasis':      ['beginning', 'middle', 'end'],
    'propositional_attitude': ['incredulity', 'neutral', 'sarcasm', 'surprise'],
}

CONDITION_COLORS = {
    # emotion
    'anger':   '#e74c3c',
    'fear':    '#9b59b6',
    'joy':     '#f39c12',
    'sadness': '#3498db',
    # topical_emphasis
    'beginning': '#e74c3c',
    'middle':    '#f39c12',
    'end':       '#2ecc71',
    # propositional_attitude
    'incredulity': '#e74c3c',
    'sarcasm':     '#9b59b6',
    'surprise':    '#f39c12',
    # shared neutral
    'neutral':  '#7f8c8d',
}

os.makedirs(PLOTS_DIR, exist_ok=True)


# ── acoustic feature extraction ───────────────────────────────────────────────

def extract_f0(wav_path, time_step=0.01, f0_min=75, f0_max=500):
    """Return (times, f0_hz) arrays with NaN for unvoiced frames."""
    snd = parselmouth.Sound(wav_path)
    pitch = snd.to_pitch(time_step=time_step, pitch_floor=f0_min, pitch_ceiling=f0_max)
    times = pitch.xs()
    f0    = pitch.selected_array['frequency']
    f0[f0 == 0] = np.nan
    return times, f0


def extract_intensity(wav_path, time_step=0.01):
    """Return (times, intensity_dB) arrays."""
    snd = parselmouth.Sound(wav_path)
    intensity = snd.to_intensity(time_step=time_step)
    times = intensity.xs()
    vals  = intensity.values.T.squeeze()
    return times, vals


def duration_seconds(wav_path):
    snd = parselmouth.Sound(wav_path)
    return snd.duration


def f0_stats(f0):
    voiced = f0[~np.isnan(f0)]
    if len(voiced) == 0:
        return dict(mean=np.nan, std=np.nan, range=np.nan, voiced_ratio=0.0)
    return dict(
        mean=float(np.mean(voiced)),
        std=float(np.std(voiced)),
        range=float(np.nanmax(f0) - np.nanmin(f0)),
        voiced_ratio=float(len(voiced) / len(f0)),
    )


# ── file discovery ────────────────────────────────────────────────────────────

def find_synth_wav(category, condition, item_id):
    """Find the HiFi-GAN synthesized wav, falling back to Griffin-Lim."""
    pattern_hifi = os.path.join(OUTPUT_ROOT, category, condition,
                                f"{item_id}_spk_{SPEAKER_ID}_ref_*_hifigan.wav")
    matches = glob.glob(pattern_hifi)
    if matches:
        return matches[0]
    pattern_gl = os.path.join(OUTPUT_ROOT, category, condition,
                              f"{item_id}_spk_{SPEAKER_ID}_ref_*.wav")
    matches = glob.glob(pattern_gl)
    return matches[0] if matches else None


def find_ref_wav(category, condition, item_id):
    path = os.path.join(ADEPT_ROOT, 'wav_44khz', category, condition, f"{item_id}.wav")
    return path if os.path.isfile(path) else None


# ── plotting ──────────────────────────────────────────────────────────────────

def plot_f0_overlay(records, title, out_path):
    """
    records = list of dicts with keys:
        label, times_ref, f0_ref, times_synth, f0_synth, color
    """
    n = len(records)
    fig, axes = plt.subplots(n, 2, figsize=(14, 2.8 * n), sharex=False)
    if n == 1:
        axes = [axes]

    fig.suptitle(title, fontsize=13, fontweight='bold')

    for ax_row, rec in zip(axes, records):
        ax_ref, ax_syn = ax_row
        color = rec['color']
        label = rec['label']

        ax_ref.plot(rec['times_ref'], rec['f0_ref'], color=color, lw=1.2)
        ax_ref.set_title(f"{label} — reference", fontsize=9)
        ax_ref.set_ylabel('F0 (Hz)', fontsize=8)
        ax_ref.set_ylim(50, 450)
        ax_ref.grid(True, alpha=0.3)

        ax_syn.plot(rec['times_synth'], rec['f0_synth'], color=color, lw=1.2, linestyle='--')
        ax_syn.set_title(f"{label} — synthesised", fontsize=9)
        ax_syn.set_ylim(50, 450)
        ax_syn.grid(True, alpha=0.3)

    for ax in axes[-1]:
        ax.set_xlabel('Time (s)', fontsize=8)

    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close()


def plot_f0_stats_bar(stats_by_condition, metric, title, out_path):
    """Bar chart comparing a scalar F0 metric across conditions for ref vs synth."""
    conditions = list(stats_by_condition.keys())
    ref_vals   = [stats_by_condition[c]['ref'][metric]   for c in conditions]
    synth_vals = [stats_by_condition[c]['synth'][metric] for c in conditions]

    x = np.arange(len(conditions))
    w = 0.35
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(x - w/2, ref_vals,   w, label='Reference', color='#2c3e50', alpha=0.8)
    ax.bar(x + w/2, synth_vals, w, label='Synthesised', color='#e67e22', alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(conditions, rotation=20, ha='right')
    ax.set_ylabel(metric)
    ax.set_title(title)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close()


def plot_duration_comparison(duration_data, title, out_path):
    """Grouped bar: speaking duration per condition, ref vs synth."""
    conditions = list(duration_data.keys())
    ref_durs   = [duration_data[c]['ref']   for c in conditions]
    synth_durs = [duration_data[c]['synth'] for c in conditions]

    x = np.arange(len(conditions))
    w = 0.35
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(x - w/2, ref_durs,   w, label='Reference', color='#2c3e50', alpha=0.8)
    ax.bar(x + w/2, synth_durs, w, label='Synthesised', color='#e67e22', alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(conditions, rotation=20, ha='right')
    ax.set_ylabel('Duration (s)')
    ax.set_title(title)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close()


# ── per-sentence overlay: all conditions on one plot ─────────────────────────

def plot_sentence_overlay(sentence_data, category, item_id, out_path):
    """
    sentence_data: list of (condition, times_ref, f0_ref, times_synth, f0_synth)
    Two-panel figure: left = all refs, right = all synths.
    """
    fig, (ax_r, ax_s) = plt.subplots(1, 2, figsize=(12, 4))
    for condition, t_r, f_r, t_s, f_s in sentence_data:
        color = CONDITION_COLORS.get(condition, '#333333')
        ax_r.plot(t_r, f_r, color=color, lw=1.4, label=condition)
        ax_s.plot(t_s, f_s, color=color, lw=1.4, linestyle='--', label=condition)

    for ax, panel in [(ax_r, 'References'), (ax_s, 'Synthesised')]:
        ax.set_title(f"{item_id} — {panel}", fontsize=10)
        ax.set_ylabel('F0 (Hz)')
        ax.set_xlabel('Time (s)')
        ax.set_ylim(50, 450)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    fig.suptitle(f"{category}  ·  {item_id}", fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close()


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    summary_rows = []  # for CSV export

    for category, conditions in CATEGORIES.items():
        cat_plot_dir = os.path.join(PLOTS_DIR, category)
        os.makedirs(cat_plot_dir, exist_ok=True)

        # collect all item_ids (use first condition's synth dir)
        first_cond_synth_dir = os.path.join(OUTPUT_ROOT, category, conditions[0])
        if not os.path.isdir(first_cond_synth_dir):
            print(f"[SKIP] no output found for {category}")
            continue

        all_item_ids = sorted(set(
            re.match(r'(ad\d+_\d+)', os.path.basename(f)).group(1)
            for f in glob.glob(os.path.join(first_cond_synth_dir, '*.wav'))
        ))

        # aggregate stats per condition across all items
        stats_by_condition = {c: {'ref': [], 'synth': []} for c in conditions}
        dur_by_condition   = {c: {'ref': [], 'synth': []} for c in conditions}

        for item_id in all_item_ids:
            sentence_data = []

            for condition in conditions:
                ref_wav   = find_ref_wav(category, condition, item_id)
                synth_wav = find_synth_wav(category, condition, item_id)
                if not ref_wav or not synth_wav:
                    continue

                t_r, f_r = extract_f0(ref_wav)
                t_s, f_s = extract_f0(synth_wav)
                sentence_data.append((condition, t_r, f_r, t_s, f_s))

                s_ref   = f0_stats(f_r)
                s_synth = f0_stats(f_s)
                stats_by_condition[condition]['ref'].append(s_ref)
                stats_by_condition[condition]['synth'].append(s_synth)

                dur_by_condition[condition]['ref'].append(duration_seconds(ref_wav))
                dur_by_condition[condition]['synth'].append(duration_seconds(synth_wav))

                summary_rows.append({
                    'category': category, 'condition': condition, 'item': item_id,
                    'ref_f0_mean': s_ref['mean'],   'synth_f0_mean': s_synth['mean'],
                    'ref_f0_std':  s_ref['std'],    'synth_f0_std':  s_synth['std'],
                    'ref_f0_range': s_ref['range'], 'synth_f0_range': s_synth['range'],
                    'ref_dur': dur_by_condition[condition]['ref'][-1],
                    'synth_dur': dur_by_condition[condition]['synth'][-1],
                })

            if sentence_data:
                plot_sentence_overlay(
                    sentence_data, category, item_id,
                    os.path.join(cat_plot_dir, f"{item_id}_f0_overlay.png")
                )

        # aggregate mean stats per condition
        agg_stats = {}
        for cond in conditions:
            refs   = stats_by_condition[cond]['ref']
            synths = stats_by_condition[cond]['synth']
            if not refs:
                continue
            agg_stats[cond] = {
                'ref':   {m: np.nanmean([r[m] for r in refs])   for m in ['mean', 'std', 'range']},
                'synth': {m: np.nanmean([s[m] for s in synths]) for m in ['mean', 'std', 'range']},
            }

        agg_dur = {}
        for cond in conditions:
            dref   = dur_by_condition[cond]['ref']
            dsynth = dur_by_condition[cond]['synth']
            if not dref:
                continue
            agg_dur[cond] = {
                'ref':   float(np.mean(dref)),
                'synth': float(np.mean(dsynth)),
            }

        if agg_stats:
            for metric in ['mean', 'std', 'range']:
                plot_f0_stats_bar(
                    agg_stats, metric,
                    f"{category} — F0 {metric} (avg across items)",
                    os.path.join(cat_plot_dir, f"f0_{metric}_bar.png")
                )

        if agg_dur:
            plot_duration_comparison(
                agg_dur,
                f"{category} — average duration",
                os.path.join(cat_plot_dir, "duration_bar.png")
            )

        print(f"[{category}] plots saved to {cat_plot_dir}/")

    # save CSV summary
    if summary_rows:
        import csv
        csv_path = os.path.join(PLOTS_DIR, 'summary.csv')
        fieldnames = list(summary_rows[0].keys())
        with open(csv_path, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(summary_rows)
        print(f"\nSummary CSV: {csv_path}")

    print(f"\nAll plots in {PLOTS_DIR}/")


if __name__ == '__main__':
    main()
