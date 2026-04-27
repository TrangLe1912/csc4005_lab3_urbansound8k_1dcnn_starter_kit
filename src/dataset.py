from __future__ import annotations

import hashlib
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import librosa
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

URBANSOUND_CLASSES = [
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


@dataclass
class SplitData:
    train_loader: DataLoader
    val_loader: DataLoader
    test_loader: DataLoader
    class_names: list[str]
    resolved_data_dir: str
    input_channels: int
    train_size: int
    val_size: int
    test_size: int


def _extract_zip_if_needed(data_path: Path) -> Path:
    if data_path.is_dir():
        return data_path
    if data_path.is_file() and data_path.suffix.lower() == '.zip':
        extract_root = data_path.parent / f'{data_path.stem}_extracted'
        marker = extract_root / '.extracted_ok'
        if not marker.exists():
            extract_root.mkdir(parents=True, exist_ok=True)
            with zipfile.ZipFile(data_path, 'r') as zf:
                zf.extractall(extract_root)
            marker.write_text('ok', encoding='utf-8')
        return extract_root
    raise FileNotFoundError(f'Không tìm thấy dữ liệu tại: {data_path}')


def _find_metadata_csv(root: Path) -> Path:
    candidates = list(root.rglob('UrbanSound8K.csv'))
    if not candidates:
        raise FileNotFoundError(
            'Không tìm thấy UrbanSound8K.csv. Hãy kiểm tra data_dir có đúng cấu trúc UrbanSound8K không.'
        )
    return candidates[0]


def _resolve_audio_path(dataset_root: Path, csv_path: Path, fold: int, filename: str) -> Path:
    candidates = [
        csv_path.parent.parent / 'audio' / f'fold{fold}' / filename,
        dataset_root / 'audio' / f'fold{fold}' / filename,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    matches = list(dataset_root.rglob(filename))
    if matches:
        return matches[0]
    raise FileNotFoundError(f'Không tìm thấy audio file: {filename}')


def _limit_per_class(df: pd.DataFrame, max_per_class: int | None, random_state: int) -> pd.DataFrame:
    if max_per_class is None or max_per_class <= 0:
        return df.reset_index(drop=True)
    parts: list[pd.DataFrame] = []
    for _, group in df.groupby('classID'):
        n = min(max_per_class, len(group))
        parts.append(group.sample(n=n, random_state=random_state))
    return pd.concat(parts, ignore_index=True).sample(frac=1.0, random_state=random_state).reset_index(drop=True)


def _load_metadata(data_dir: str | Path) -> tuple[pd.DataFrame, Path]:
    root = _extract_zip_if_needed(Path(data_dir))
    csv_path = _find_metadata_csv(root)
    df = pd.read_csv(csv_path)
    required = {'slice_file_name', 'fold', 'classID', 'class'}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f'UrbanSound8K.csv thiếu cột: {sorted(missing)}')

    df = df.copy()
    df['audio_path'] = [
        str(_resolve_audio_path(root, csv_path, int(row.fold), str(row.slice_file_name)))
        for row in df.itertuples(index=False)
    ]
    df['fold'] = df['fold'].astype(int)
    df['classID'] = df['classID'].astype(int)
    return df, root


def _safe_normalize_feature(x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    mean = x.mean(axis=-1, keepdims=True)
    std = x.std(axis=-1, keepdims=True)
    return (x - mean) / (std + eps)


def _pad_or_crop(y: np.ndarray, target_len: int, random_crop: bool = False) -> np.ndarray:
    if len(y) == target_len:
        return y
    if len(y) > target_len:
        if random_crop:
            start = int(np.random.randint(0, len(y) - target_len + 1))
        else:
            start = max(0, (len(y) - target_len) // 2)
        return y[start:start + target_len]
    out = np.zeros(target_len, dtype=np.float32)
    out[:len(y)] = y
    return out


def _time_freq_mask(x: np.ndarray, max_time_width: int = 10, max_freq_width: int = 5) -> np.ndarray:
    out = x.copy()
    if out.shape[1] > 2 and max_time_width > 0:
        width = int(np.random.randint(0, min(max_time_width, out.shape[1]) + 1))
        if width > 0:
            start = int(np.random.randint(0, out.shape[1] - width + 1))
            out[:, start:start + width] = 0.0
    if out.shape[0] > 2 and max_freq_width > 0:
        width = int(np.random.randint(0, min(max_freq_width, out.shape[0]) + 1))
        if width > 0:
            start = int(np.random.randint(0, out.shape[0] - width + 1))
            out[start:start + width, :] = 0.0
    return out


class UrbanSoundFeatureDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        feature_type: str = 'mfcc',
        sample_rate: int = 16000,
        duration: float = 4.0,
        n_fft: int = 1024,
        hop_length: int = 512,
        n_mfcc: int = 40,
        n_mels: int = 64,
        augment: bool = False,
        cache_dir: str | Path | None = '.cache/features',
    ):
        self.df = df.reset_index(drop=True)
        self.feature_type = feature_type
        self.sample_rate = int(sample_rate)
        self.duration = float(duration)
        self.n_fft = int(n_fft)
        self.hop_length = int(hop_length)
        self.n_mfcc = int(n_mfcc)
        self.n_mels = int(n_mels)
        self.augment = augment
        self.target_len = int(self.sample_rate * self.duration)
        self.cache_dir = Path(cache_dir) if cache_dir else None
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    def __len__(self) -> int:
        return len(self.df)

    @property
    def input_channels(self) -> int:
        if self.feature_type == 'mfcc':
            return self.n_mfcc
        if self.feature_type == 'logmel':
            return self.n_mels
        if self.feature_type == 'raw':
            return 1
        raise ValueError(f'Unsupported feature_type: {self.feature_type}')

    def _cache_path(self, audio_path: Path) -> Path | None:
        if self.cache_dir is None:
            return None
        key_data = {
            'path': str(audio_path.resolve()),
            'feature_type': self.feature_type,
            'sample_rate': self.sample_rate,
            'duration': self.duration,
            'n_fft': self.n_fft,
            'hop_length': self.hop_length,
            'n_mfcc': self.n_mfcc,
            'n_mels': self.n_mels,
        }
        key = hashlib.sha1(str(key_data).encode('utf-8')).hexdigest()[:20]
        return self.cache_dir / f'{key}.npy'

    def _load_waveform(self, audio_path: Path) -> np.ndarray:
        y, _ = librosa.load(audio_path, sr=self.sample_rate, mono=True)
        y = y.astype(np.float32)
        y = _pad_or_crop(y, self.target_len, random_crop=self.augment and self.feature_type == 'raw')
        peak = np.max(np.abs(y))
        if peak > 0:
            y = y / peak
        return y

    def _compute_feature(self, audio_path: Path) -> np.ndarray:
        y = self._load_waveform(audio_path)
        if self.feature_type == 'raw':
            return y[None, :].astype(np.float32)
        if self.feature_type == 'mfcc':
            feat = librosa.feature.mfcc(
                y=y,
                sr=self.sample_rate,
                n_mfcc=self.n_mfcc,
                n_fft=self.n_fft,
                hop_length=self.hop_length,
            )
            feat = _safe_normalize_feature(feat)
            return feat.astype(np.float32)
        if self.feature_type == 'logmel':
            mel = librosa.feature.melspectrogram(
                y=y,
                sr=self.sample_rate,
                n_fft=self.n_fft,
                hop_length=self.hop_length,
                n_mels=self.n_mels,
                power=2.0,
            )
            feat = librosa.power_to_db(mel, ref=np.max)
            feat = _safe_normalize_feature(feat)
            return feat.astype(np.float32)
        raise ValueError(f'Unsupported feature_type: {self.feature_type}')

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        row = self.df.iloc[idx]
        audio_path = Path(row['audio_path'])
        label = int(row['classID'])
        cache_path = self._cache_path(audio_path)
        if cache_path is not None and cache_path.exists():
            feat = np.load(cache_path)
        else:
            feat = self._compute_feature(audio_path)
            if cache_path is not None:
                np.save(cache_path, feat)

        if self.augment:
            if self.feature_type in {'mfcc', 'logmel'}:
                feat = _time_freq_mask(feat)
            elif self.feature_type == 'raw':
                shift = int(np.random.uniform(-0.05, 0.05) * feat.shape[-1])
                feat = np.roll(feat, shift, axis=-1)
                noise = np.random.normal(0.0, 0.005, size=feat.shape).astype(np.float32)
                feat = feat + noise

        return torch.from_numpy(feat.astype(np.float32)), label


def create_dataloaders(
    data_dir: str | Path,
    feature_type: str = 'mfcc',
    sample_rate: int = 16000,
    duration: float = 4.0,
    n_fft: int = 1024,
    hop_length: int = 512,
    n_mfcc: int = 40,
    n_mels: int = 64,
    batch_size: int = 32,
    train_folds: list[int] | None = None,
    val_folds: list[int] | None = None,
    test_folds: list[int] | None = None,
    max_train_per_class: int | None = 120,
    max_eval_per_class: int | None = 50,
    random_state: int = 42,
    augment: bool = False,
    cache_dir: str | Path | None = '.cache/features',
    num_workers: int = 0,
) -> SplitData:
    train_folds = train_folds or [1, 2, 3, 4, 5, 6, 7, 8]
    val_folds = val_folds or [9]
    test_folds = test_folds or [10]

    df, root = _load_metadata(data_dir)
    train_df = df[df['fold'].isin(train_folds)].copy()
    val_df = df[df['fold'].isin(val_folds)].copy()
    test_df = df[df['fold'].isin(test_folds)].copy()

    train_df = _limit_per_class(train_df, max_train_per_class, random_state)
    val_df = _limit_per_class(val_df, max_eval_per_class, random_state)
    test_df = _limit_per_class(test_df, max_eval_per_class, random_state)

    if train_df.empty or val_df.empty or test_df.empty:
        raise ValueError('Một trong các split train/val/test đang rỗng. Hãy kiểm tra cấu hình folds.')

    class_map = df[['classID', 'class']].drop_duplicates().sort_values('classID')
    class_names = class_map['class'].tolist()

    common_kwargs: dict[str, Any] = dict(
        feature_type=feature_type,
        sample_rate=sample_rate,
        duration=duration,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mfcc=n_mfcc,
        n_mels=n_mels,
        cache_dir=cache_dir,
    )
    train_ds = UrbanSoundFeatureDataset(train_df, augment=augment, **common_kwargs)
    val_ds = UrbanSoundFeatureDataset(val_df, augment=False, **common_kwargs)
    test_ds = UrbanSoundFeatureDataset(test_df, augment=False, **common_kwargs)

    return SplitData(
        train_loader=DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers),
        val_loader=DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers),
        test_loader=DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers),
        class_names=class_names,
        resolved_data_dir=str(root),
        input_channels=train_ds.input_channels,
        train_size=len(train_ds),
        val_size=len(val_ds),
        test_size=len(test_ds),
    )
