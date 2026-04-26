import os
import glob
import whisper
import jiwer
import re
from pathlib import Path

def normalize_text(text):
    """Normalize text by removing punctuation and standardizing whitespace."""
    # Remove punctuation and extra whitespace, convert to lowercase
    text = re.sub(r'[^\w\s]', '', text.lower())
    return ' '.join(text.split())

def get_ground_truth_text(wav_path):
    """Get ground truth text for a given WAV file path."""
    # Convert path from adept_output/emotion/anger/ad00_0000_spk_0_ref_ad00_0000_hifigan.wav
    # to ADEPT/txt/emotion/anger/ad00_0000.txt

    parts = wav_path.split('/')
    # Expected: ['', 'root', 'adept_output', 'emotion', 'anger', 'ad00_0000_..._hifigan.wav']
    if len(parts) < 6:
        return None

    category = parts[3]    # emotion, propositional_attitude, etc.
    subcategory = parts[4] # anger, fear, beginning, etc.
    filename = parts[5]

    # Extract the base name (ad00_0000)
    base_name = filename.split('_')[0] + '_' + filename.split('_')[1]

    # Construct text file path
    txt_path = f'/root/ADEPT/txt/{category}/{subcategory}/{base_name}.txt'

    if os.path.exists(txt_path):
        with open(txt_path, 'r', encoding='utf-8') as f:
            return f.read().strip()
    else:
        return None

def main():
    print('Loading Whisper model...')
    model = whisper.load_model('base')

    # Find all HiFi-GAN WAV files
    wav_files = glob.glob('/root/adept_output/*/*/*_hifigan.wav')
    wav_files.sort()

    print(f'Found {len(wav_files)} HiFi-GAN WAV files to process')
    print('=' * 80)

    results = []

    for i, wav_file in enumerate(wav_files):
        print(f'[{i+1}/{len(wav_files)}] Processing: {os.path.basename(wav_file)}')

        # Get ground truth text
        ground_truth = get_ground_truth_text(wav_file)
        if ground_truth is None:
            print(f'  ERROR: Could not find ground truth text for {wav_file}')
            continue

        print(f'  Ground truth: "{ground_truth}"')

        try:
            # Transcribe audio
            result = model.transcribe(wav_file, fp16=False)
            transcription = result['text'].strip()

            print(f'  Transcription: "{transcription}"')

            # Normalize texts
            norm_ground_truth = normalize_text(ground_truth)
            norm_transcription = normalize_text(transcription)

            print(f'  Normalized GT: "{norm_ground_truth}"')
            print(f'  Normalized TX: "{norm_transcription}"')

            # Calculate metrics
            wer = jiwer.wer(norm_ground_truth, norm_transcription)
            cer = jiwer.cer(norm_ground_truth, norm_transcription)

            print(f'  WER: {wer:.3f} ({wer*100:.1f}%)')
            print(f'  CER: {cer:.3f} ({cer*100:.1f}%)')

            # Store results
            category = wav_file.split('/')[3] + '/' + wav_file.split('/')[4]
            results.append({
                'file': os.path.basename(wav_file),
                'category': category,
                'ground_truth': ground_truth,
                'transcription': transcription,
                'wer': wer,
                'cer': cer
            })

        except Exception as e:
            print(f'  ERROR: {str(e)}')

        print()

    # Summary statistics
    if results:
        print('SUMMARY STATISTICS')
        print('=' * 80)

        categories = {}
        for result in results:
            cat = result['category']
            if cat not in categories:
                categories[cat] = []
            categories[cat].append(result['wer'])

        for cat, wers in categories.items():
            avg_wer = sum(wers) / len(wers)
            print(f'{cat}: {len(wers)} files, Avg WER: {avg_wer:.3f} ({avg_wer*100:.1f}%)')

        overall_avg_wer = sum(r['wer'] for r in results) / len(results)
        print(f'\nOverall: {len(results)} files, Avg WER: {overall_avg_wer:.3f} ({overall_avg_wer*100:.1f}%)')

        # Save detailed results
        with open('/root/adept_wer_results.txt', 'w') as f:
            f.write('File\tCategory\tGround Truth\tTranscription\tWER\tCER\n')
            for result in results:
                f.write(f"{result['file']}\t{result['category']}\t{result['ground_truth']}\t{result['transcription']}\t{result['wer']:.3f}\t{result['cer']:.3f}\n")

        print(f'\nDetailed results saved to: /root/adept_wer_results.txt')

if __name__ == '__main__':
    main()