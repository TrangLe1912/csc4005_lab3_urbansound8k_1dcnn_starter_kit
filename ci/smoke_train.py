from __future__ import annotations

import csv
import math
import shutil
import subprocess
import sys
import wave
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
TMP = ROOT / '.tmp_ci_urbansound'
CACHE = ROOT / '.tmp_ci_cache'

for path in [TMP, CACHE, ROOT / 'outputs' / 'ci_smoke']:
    if path.exists():
        shutil.rmtree(path)

metadata_dir = TMP / 'metadata'
audio_root = TMP / 'audio'
metadata_dir.mkdir(parents=True, exist_ok=True)

classes = [
    'air_conditioner',
    'car_horn',
    'children_playing',
    'dog_bark',
    'drilling',
    'engine_idling',
    'gun_shot',
    'jackhammer',
    'siren',
    'street_music',
]

sample_rate = 8000
duration = 0.5
num_samples = int(sample_rate * duration)
rng = np.random.default_rng(42)


def write_wav(path: Path, freq: float) -> None:
    t = np.arange(num_samples, dtype=np.float32) / sample_rate
    y = 0.4 * np.sin(2 * math.pi * freq * t)
    y += 0.03 * rng.normal(size=num_samples)
    y = np.clip(y, -1.0, 1.0)
    pcm = (y * 32767).astype(np.int16)
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), 'wb') as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(sample_rate)
        f.writeframes(pcm.tobytes())

rows = []
for class_id, class_name in enumerate(classes):
    for fold in [1, 9, 10]:
        for idx in range(2):
            filename = f'{class_id}-{fold}-{idx}.wav'
            freq = 180 + class_id * 35 + idx * 5
            write_wav(audio_root / f'fold{fold}' / filename, freq)
            rows.append({
                'slice_file_name': filename,
                'fsID': class_id,
                'start': 0,
                'end': duration,
                'salience': 1,
                'fold': fold,
                'classID': class_id,
                'class': class_name,
            })

with (metadata_dir / 'UrbanSound8K.csv').open('w', newline='', encoding='utf-8') as f:
    writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    writer.writeheader()
    writer.writerows(rows)

cmd = [
    sys.executable,
    '-m',
    'src.train',
    '--data_dir', str(TMP),
    '--run_name', 'ci_smoke',
    '--model_name', 'mfcc_1dcnn',
    '--feature_type', 'mfcc',
    '--sample_rate', '8000',
    '--duration', '0.5',
    '--n_fft', '512',
    '--hop_length', '256',
    '--n_mfcc', '16',
    '--hidden_channels', '16,32,32',
    '--epochs', '1',
    '--batch_size', '8',
    '--train_folds', '1',
    '--val_folds', '9',
    '--test_folds', '10',
    '--max_train_per_class', '2',
    '--max_eval_per_class', '2',
    '--augment', 'false',
    '--cache_dir', str(CACHE),
    '--use_wandb', 'false',
    '--device', 'cpu',
]
subprocess.run(cmd, cwd=ROOT, check=True)
print('Smoke train OK')
