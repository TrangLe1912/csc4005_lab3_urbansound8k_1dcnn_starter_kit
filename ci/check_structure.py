from pathlib import Path
import sys

REQUIRED = [
    'README.md',
    'REPORT_TEMPLATE.md',
    'requirements.txt',
    'src/__init__.py',
    'src/dataset.py',
    'src/model.py',
    'src/train.py',
    'src/utils.py',
    'docs/LAB_GUIDE_LAB3.md',
    'docs/WANDB_GUIDE.md',
    'docs/GITHUB_CLASSROOM_GUIDE.md',
    'configs/baseline_mfcc_1dcnn.json',
    'configs/fast_debug.json',
    'configs/extension_raw_waveform.json',
    'ci/smoke_train.py',
    '.github/workflows/validate-lab3.yml',
]

root = Path(__file__).resolve().parents[1]
missing = [p for p in REQUIRED if not (root / p).exists()]
if missing:
    print('Missing files:')
    for item in missing:
        print('-', item)
    sys.exit(1)
print('Structure OK')
