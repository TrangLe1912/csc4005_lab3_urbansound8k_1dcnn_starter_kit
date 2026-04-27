from __future__ import annotations

import argparse
import os
import time
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.optim import AdamW, SGD
from torch.optim.lr_scheduler import ReduceLROnPlateau

from src.dataset import create_dataloaders
from src.model import build_model
from src.utils import (
    EarlyStopping,
    classification_report_dict,
    compute_accuracy,
    count_parameters,
    ensure_dir,
    load_json,
    plot_curves,
    save_confusion_matrix,
    save_history_csv,
    save_json,
    set_seed,
)

try:
    import wandb
except ImportError:  # pragma: no cover
    wandb = None


def str2bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    value = value.lower().strip()
    if value in {'true', '1', 'yes', 'y'}:
        return True
    if value in {'false', '0', 'no', 'n'}:
        return False
    raise argparse.ArgumentTypeError(f'Không đọc được boolean value: {value}')


def parse_int_list(value: Any) -> list[int]:
    if isinstance(value, list):
        return [int(v) for v in value]
    if isinstance(value, tuple):
        return [int(v) for v in value]
    if isinstance(value, str):
        return [int(v.strip()) for v in value.split(',') if v.strip()]
    raise argparse.ArgumentTypeError(f'Không đọc được list fold: {value}')


def parse_hidden_channels(value: Any) -> list[int]:
    return parse_int_list(value)


def load_config_from_argv() -> dict[str, Any]:
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument('--config', type=str, default=None)
    known, _ = pre.parse_known_args()
    if known.config:
        return load_json(known.config)
    return {}


def parse_args() -> argparse.Namespace:
    cfg = load_config_from_argv()
    parser = argparse.ArgumentParser(description='Train 1D-CNN for UrbanSound8K classification')
    parser.add_argument('--config', type=str, default=None, help='Đường dẫn file JSON cấu hình')
    parser.add_argument('--data_dir', type=str, required=True, help='Đường dẫn tới UrbanSound8K folder hoặc UrbanSound8K.zip')
    parser.add_argument('--project', type=str, default=cfg.get('project', 'csc4005-lab3-urbansound-1dcnn'))
    parser.add_argument('--run_name', type=str, default=cfg.get('run_name', 'debug_run'))
    parser.add_argument('--model_name', type=str, choices=['mfcc_1dcnn', 'logmel_1dcnn', 'feature_1dcnn', 'raw_1dcnn'], default=cfg.get('model_name', 'mfcc_1dcnn'))
    parser.add_argument('--feature_type', type=str, choices=['mfcc', 'logmel', 'raw'], default=cfg.get('feature_type', 'mfcc'))
    parser.add_argument('--sample_rate', type=int, default=cfg.get('sample_rate', 16000))
    parser.add_argument('--duration', type=float, default=cfg.get('duration', 4.0))
    parser.add_argument('--n_fft', type=int, default=cfg.get('n_fft', 1024))
    parser.add_argument('--hop_length', type=int, default=cfg.get('hop_length', 512))
    parser.add_argument('--n_mfcc', type=int, default=cfg.get('n_mfcc', 40))
    parser.add_argument('--n_mels', type=int, default=cfg.get('n_mels', 64))
    parser.add_argument('--hidden_channels', type=parse_hidden_channels, default=cfg.get('hidden_channels', [64, 128, 128]))
    parser.add_argument('--dropout', type=float, default=cfg.get('dropout', 0.35))
    parser.add_argument('--optimizer', type=str, choices=['adamw', 'sgd'], default=cfg.get('optimizer', 'adamw'))
    parser.add_argument('--scheduler', type=str, choices=['none', 'plateau'], default=cfg.get('scheduler', 'plateau'))
    parser.add_argument('--lr', type=float, default=cfg.get('lr', 1e-3))
    parser.add_argument('--weight_decay', type=float, default=cfg.get('weight_decay', 1e-4))
    parser.add_argument('--epochs', type=int, default=cfg.get('epochs', 12))
    parser.add_argument('--batch_size', type=int, default=cfg.get('batch_size', 32))
    parser.add_argument('--patience', type=int, default=cfg.get('patience', 4))
    parser.add_argument('--seed', type=int, default=cfg.get('seed', 42))
    parser.add_argument('--train_folds', type=parse_int_list, default=cfg.get('train_folds', [1, 2, 3, 4, 5, 6, 7, 8]))
    parser.add_argument('--val_folds', type=parse_int_list, default=cfg.get('val_folds', [9]))
    parser.add_argument('--test_folds', type=parse_int_list, default=cfg.get('test_folds', [10]))
    parser.add_argument('--max_train_per_class', type=int, default=cfg.get('max_train_per_class', 120))
    parser.add_argument('--max_eval_per_class', type=int, default=cfg.get('max_eval_per_class', 50))
    parser.add_argument('--augment', type=str2bool, default=cfg.get('augment', True))
    parser.add_argument('--cache_dir', type=str, default=cfg.get('cache_dir', '.cache/features'))
    parser.add_argument('--num_workers', type=int, default=cfg.get('num_workers', 0))
    parser.add_argument('--device', type=str, choices=['auto', 'cpu', 'cuda', 'mps'], default=cfg.get('device', 'auto'))
    parser.add_argument('--use_wandb', type=str2bool, default=cfg.get('use_wandb', True))
    parser.add_argument('--wandb_mode', type=str, choices=['online', 'offline', 'disabled'], default=cfg.get('wandb_mode', 'online'))
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.feature_type == 'raw' and args.model_name != 'raw_1dcnn':
        raise ValueError('feature_type=raw cần dùng model_name=raw_1dcnn.')
    if args.feature_type in {'mfcc', 'logmel'} and args.model_name == 'raw_1dcnn':
        raise ValueError('raw_1dcnn chỉ dùng với feature_type=raw.')
    if args.feature_type == 'mfcc' and args.model_name == 'logmel_1dcnn':
        raise ValueError('model_name=logmel_1dcnn nên đi với feature_type=logmel.')
    if args.feature_type == 'logmel' and args.model_name == 'mfcc_1dcnn':
        args.model_name = 'logmel_1dcnn'
    if args.wandb_mode == 'disabled':
        args.use_wandb = False


def select_device(name: str) -> torch.device:
    if name == 'cpu':
        return torch.device('cpu')
    if name == 'cuda':
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if name == 'mps':
        return torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    if torch.cuda.is_available():
        return torch.device('cuda')
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


def get_optimizer(name: str, model: nn.Module, lr: float, weight_decay: float):
    params = [p for p in model.parameters() if p.requires_grad]
    if name == 'adamw':
        return AdamW(params, lr=lr, weight_decay=weight_decay)
    if name == 'sgd':
        return SGD(params, lr=lr, momentum=0.9, weight_decay=weight_decay)
    raise ValueError(f'Unsupported optimizer: {name}')


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    y_true, y_pred = [], []
    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * x.size(0)
        preds = torch.argmax(logits, dim=1)
        y_true.extend(y.detach().cpu().tolist())
        y_pred.extend(preds.detach().cpu().tolist())
    return running_loss / len(loader.dataset), compute_accuracy(y_true, y_pred)


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    y_true, y_pred = [], []
    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        logits = model(x)
        loss = criterion(logits, y)
        running_loss += loss.item() * x.size(0)
        preds = torch.argmax(logits, dim=1)
        y_true.extend(y.detach().cpu().tolist())
        y_pred.extend(preds.detach().cpu().tolist())
    return running_loss / len(loader.dataset), compute_accuracy(y_true, y_pred), y_true, y_pred


def main() -> None:
    args = parse_args()
    validate_args(args)
    set_seed(args.seed)
    torch.set_num_threads(max(1, min(4, os.cpu_count() or 1)))
    device = select_device(args.device)
    output_dir = ensure_dir(Path('outputs') / args.run_name)

    data = create_dataloaders(
        data_dir=args.data_dir,
        feature_type=args.feature_type,
        sample_rate=args.sample_rate,
        duration=args.duration,
        n_fft=args.n_fft,
        hop_length=args.hop_length,
        n_mfcc=args.n_mfcc,
        n_mels=args.n_mels,
        batch_size=args.batch_size,
        train_folds=args.train_folds,
        val_folds=args.val_folds,
        test_folds=args.test_folds,
        max_train_per_class=args.max_train_per_class,
        max_eval_per_class=args.max_eval_per_class,
        random_state=args.seed,
        augment=args.augment,
        cache_dir=args.cache_dir,
        num_workers=args.num_workers,
    )

    print(f'Resolved data directory: {data.resolved_data_dir}')
    print(f'Classes: {data.class_names}')
    print(f'Split sizes: train={data.train_size}, val={data.val_size}, test={data.test_size}')
    print(f'Input channels: {data.input_channels}')
    print(f'Device: {device}')

    model = build_model(
        model_name=args.model_name,
        input_channels=data.input_channels,
        num_classes=len(data.class_names),
        hidden_channels=args.hidden_channels,
        dropout=args.dropout,
    ).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = get_optimizer(args.optimizer, model, args.lr, args.weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2) if args.scheduler == 'plateau' else None
    total_params, trainable_params = count_parameters(model)

    used_config = vars(args).copy()
    used_config.update({
        'num_classes': len(data.class_names),
        'class_names': data.class_names,
        'device_resolved': str(device),
        'resolved_data_dir': data.resolved_data_dir,
        'input_channels': data.input_channels,
        'train_size': data.train_size,
        'val_size': data.val_size,
        'test_size': data.test_size,
        'total_params': total_params,
        'trainable_params': trainable_params,
    })
    save_json(used_config, output_dir / 'used_config.json')

    use_wandb = args.use_wandb
    if use_wandb and wandb is None:
        raise RuntimeError('Bạn đang bật W&B nhưng chưa cài wandb. Hãy chạy: pip install wandb')
    if use_wandb:
        wandb.init(project=args.project, name=args.run_name, config=used_config, mode=args.wandb_mode)

    history: list[dict[str, float]] = []
    early_stopper = EarlyStopping(patience=args.patience)
    best_val_loss = float('inf')
    best_val_acc = 0.0

    for epoch in range(1, args.epochs + 1):
        start = time.perf_counter()
        train_loss, train_acc = train_one_epoch(model, data.train_loader, criterion, optimizer, device)
        val_loss, val_acc, _, _ = evaluate(model, data.val_loader, criterion, device)
        if scheduler is not None:
            scheduler.step(val_loss)
        epoch_time = time.perf_counter() - start
        lr_current = optimizer.param_groups[0]['lr']
        row = {
            'epoch': epoch,
            'train_loss': round(train_loss, 6),
            'train_acc': round(train_acc, 6),
            'val_loss': round(val_loss, 6),
            'val_acc': round(val_acc, 6),
            'lr': lr_current,
            'epoch_time_sec': round(epoch_time, 4),
        }
        history.append(row)
        print(
            f"Epoch {epoch:02d}/{args.epochs} | train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} | lr={lr_current:.6f} | sec={epoch_time:.2f}"
        )
        if use_wandb:
            wandb.log(row)
        if early_stopper.step(val_loss):
            best_val_loss = val_loss
            best_val_acc = val_acc
            torch.save(model.state_dict(), output_dir / 'best_model.pt')
        if early_stopper.should_stop:
            print(f'Early stopping at epoch {epoch}')
            break

    if not (output_dir / 'best_model.pt').exists():
        torch.save(model.state_dict(), output_dir / 'best_model.pt')

    model.load_state_dict(torch.load(output_dir / 'best_model.pt', map_location=device))
    test_loss, test_acc, y_true, y_pred = evaluate(model, data.test_loader, criterion, device)
    report = classification_report_dict(y_true, y_pred, data.class_names)
    cm = save_confusion_matrix(y_true, y_pred, data.class_names, output_dir / 'confusion_matrix.png')
    plot_curves(history, output_dir / 'curves.png')
    save_history_csv(history, output_dir / 'history.csv')
    avg_epoch_time = sum(row['epoch_time_sec'] for row in history) / max(len(history), 1)
    metrics = {
        'model_name': args.model_name,
        'feature_type': args.feature_type,
        'best_val_loss': best_val_loss,
        'best_val_acc': best_val_acc,
        'test_loss': test_loss,
        'test_acc': test_acc,
        'avg_epoch_time_sec': avg_epoch_time,
        'total_params': total_params,
        'trainable_params': trainable_params,
        'class_names': data.class_names,
        'classification_report': report,
        'confusion_matrix': cm.tolist(),
        'resolved_data_dir': data.resolved_data_dir,
    }
    save_json(metrics, output_dir / 'metrics.json')

    print(f'Best val acc: {best_val_acc:.4f}')
    print(f'Test acc: {test_acc:.4f}')
    print(f'Average epoch time: {avg_epoch_time:.2f} sec')
    print(f'Trainable params: {trainable_params:,}')
    print(f'Saved outputs to: {output_dir}')

    if use_wandb:
        wandb.log({
            'best_val_acc': best_val_acc,
            'best_val_loss': best_val_loss,
            'test_acc': test_acc,
            'test_loss': test_loss,
            'avg_epoch_time_sec': avg_epoch_time,
            'trainable_params': trainable_params,
            'total_params': total_params,
            'confusion_matrix_image': wandb.Image(str(output_dir / 'confusion_matrix.png')),
            'curves_image': wandb.Image(str(output_dir / 'curves.png')),
        })
        wandb.finish()


if __name__ == '__main__':
    main()
