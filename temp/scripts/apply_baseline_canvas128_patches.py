#!/usr/bin/env python3
"""Re-apply canvas128-subset patches onto fresh iTransformer / PatchTST clones.

Idempotent. Fail-fast if clones are missing when invoked from the runner.
"""
from __future__ import annotations

from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
ITRANS_FACTORY = REPO / "temp/iTransformer/data_provider/data_factory.py"
PATCH_FACTORY = REPO / "temp/PatchTST/PatchTST_supervised/data_provider/data_factory.py"
ITRANS_RUN = REPO / "temp/iTransformer/run.py"
PATCH_RUN = REPO / "temp/PatchTST/PatchTST_supervised/run_longExp.py"
ITRANS_ATTN = REPO / "temp/iTransformer/layers/SelfAttention_Family.py"
ITRANS_EXP = REPO / "temp/iTransformer/experiments/exp_long_term_forecasting.py"
PATCH_EXP = REPO / "temp/PatchTST/PatchTST_supervised/exp/exp_main.py"

# PeMS CSV: 60/20/20 like Dataset_PEMS / repo _paper_split_borders; zero time marks.
_PEMS_CSV_CLASS = '''
class Dataset_PEMS_CSV(Dataset):
    """CSV PeMS with 60/20/20 split (not Dataset_Custom 70/10/20)."""

    def __init__(self, root_path, flag='train', size=None,
                 features='S', data_path='PeMS.csv',
                 target='OT', scale=True, timeenc=0, freq='h'):
        self.seq_len = size[0]
        self.label_len = size[1]
        self.pred_len = size[2]
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]
        self.features = features
        self.target = target
        self.scale = scale
        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        from sklearn.preprocessing import StandardScaler
        import pandas as pd
        import os
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path, self.data_path))
        cols = list(df_raw.columns)
        cols.remove(self.target)
        cols.remove('date')
        df_raw = df_raw[['date'] + cols + [self.target]]
        num_train = int(len(df_raw) * 0.6)
        num_vali = int(len(df_raw) * 0.2)
        num_test = len(df_raw) - num_train - num_vali
        border1s = [0, num_train - self.seq_len, len(df_raw) - num_test - self.seq_len]
        border2s = [num_train, num_train + num_vali, len(df_raw)]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]
        cols_data = df_raw.columns[1:]
        df_data = df_raw[cols_data]
        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values
        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]

    def __getitem__(self, index):
        import torch
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len
        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = torch.zeros((seq_x.shape[0], 1))
        seq_y_mark = torch.zeros((seq_y.shape[0], 1))
        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)
'''

FACTORY_BODY_ITRANS = '''from data_provider.data_loader import Dataset_ETT_hour, Dataset_ETT_minute, Dataset_Custom, Dataset_Solar, Dataset_PEMS, \\
    Dataset_Pred
from torch.utils.data import DataLoader, Dataset
''' + _PEMS_CSV_CLASS + '''

data_dict = {
    'ETTh1': Dataset_ETT_hour,
    'ETTh2': Dataset_ETT_hour,
    'ETTm1': Dataset_ETT_minute,
    'ETTm2': Dataset_ETT_minute,
    'Solar': Dataset_Solar,
    'PEMS': Dataset_PEMS_CSV,
    'custom': Dataset_Custom,
}


class _StrideWrap(Dataset):
    def __init__(self, base, stride: int):
        self.base = base
        self.stride = max(1, int(stride))
        self.indices = list(range(0, len(base), self.stride))

    def __getattr__(self, name):
        return getattr(self.base, name)

    def __getitem__(self, index):
        return self.base[self.indices[index]]

    def __len__(self):
        return len(self.indices)


class _WindowCapWrap(Dataset):
    def __init__(self, base, max_windows: int, seed: int):
        import random
        self.base = base
        n = len(base)
        k = int(max_windows)
        if k < 1:
            raise ValueError('max_windows must be >= 1, got %r' % (max_windows,))
        if k >= n:
            self.indices = list(range(n))
        else:
            rng = random.Random(int(seed))
            self.indices = sorted(rng.sample(range(n), k))

    def __getattr__(self, name):
        return getattr(self.base, name)

    def __getitem__(self, index):
        return self.base[self.indices[index]]

    def __len__(self):
        return len(self.indices)


class _IndexWrap(Dataset):
    def __init__(self, base, indices):
        self.base = base
        self.indices = list(indices)
        n = len(base)
        if not self.indices:
            raise ValueError('index wrap is empty')
        if any(i < 0 or i >= n for i in self.indices):
            raise ValueError('index wrap out of range for n=%d' % n)

    def __getattr__(self, name):
        return getattr(self.base, name)

    def __getitem__(self, index):
        return self.base[self.indices[index]]

    def __len__(self):
        return len(self.indices)


def _apply_window_cap(data_set, args, flag):
    if flag == 'pred':
        return data_set
    seed = int(getattr(args, 'window_subset_seed', 42))
    if flag == 'train':
        cap = int(getattr(args, 'train_max_windows', 0) or 0)
        seed = seed + 17
    elif flag == 'val':
        cap = int(getattr(args, 'val_max_windows', 0) or 0)
        seed = seed + 29
    else:
        cap = int(getattr(args, 'test_max_windows', 0) or 0)
        frac = float(getattr(args, 'eval_test_fraction', 0) or 0)
        ref_pred = int(getattr(args, 'eval_ref_pred_len', 0) or 0)
        if cap > 0 and 0.0 < frac < 1.0:
            raise ValueError('test_max_windows and eval_test_fraction cannot both apply')
        if cap < 1 and 0.0 < frac < 1.0 and ref_pred > 0:
            import random
            pred_len = int(args.pred_len)
            stride = 1
            unstrided = data_set
            if isinstance(data_set, _StrideWrap):
                stride = int(data_set.stride)
                unstrided = data_set.base
            n_unstrided_ref = len(unstrided) + pred_len - ref_pred
            if n_unstrided_ref < 1:
                raise ValueError('eval_ref_pred_len=%s leaves no campaign-H windows' % ref_pred)
            n_ref = len(range(0, n_unstrided_ref, stride))
            k = max(1, int(round(n_ref * frac)))
            if k >= n_ref:
                idxs = list(range(n_ref))
            else:
                rng = random.Random(int(seed))
                idxs = sorted(rng.sample(range(n_ref), k))
            n = len(data_set)
            idxs = [i for i in idxs if i < n]
            if not idxs:
                raise ValueError('binary_10pct empty after clip to pred_len=%s' % pred_len)
            return _IndexWrap(data_set, idxs)
        if cap < 1 and 0.0 < frac < 1.0:
            n = len(data_set)
            cap = max(1, int(round(n * frac)))
    if cap > 0:
        data_set = _WindowCapWrap(data_set, cap, seed)
    return data_set


def data_provider(args, flag):
    Data = data_dict[args.data]
    timeenc = 0 if args.embed != 'timeF' else 1

    if flag == 'test':
        shuffle_flag = False
        drop_last = False
        batch_size = 1
        freq = args.freq
    elif flag == 'pred':
        shuffle_flag = False
        drop_last = False
        batch_size = 1
        freq = args.freq
        Data = Dataset_Pred
    else:
        shuffle_flag = True
        drop_last = True
        batch_size = args.batch_size
        freq = args.freq

    data_set = Data(
        root_path=args.root_path,
        data_path=args.data_path,
        flag=flag,
        size=[args.seq_len, args.label_len, args.pred_len],
        features=args.features,
        target=args.target,
        timeenc=timeenc,
        freq=freq,
    )
    if flag == 'train':
        stride = int(getattr(args, 'train_window_stride', 1) or 1)
    elif flag == 'val':
        stride = int(getattr(args, 'val_window_stride', 1) or 1)
    else:
        stride = int(getattr(args, 'test_window_stride', 1) or 1)
    if stride > 1 and flag != 'pred':
        data_set = _StrideWrap(data_set, stride)
    data_set = _apply_window_cap(data_set, args, flag)
    print(flag, len(data_set), f'stride={stride}')
    data_loader = DataLoader(
        data_set,
        batch_size=batch_size,
        shuffle=shuffle_flag,
        num_workers=args.num_workers,
        drop_last=drop_last)
    return data_set, data_loader
'''

FACTORY_BODY_PATCH = '''from data_provider.data_loader import Dataset_ETT_hour, Dataset_ETT_minute, Dataset_Custom, Dataset_Pred
from torch.utils.data import DataLoader, Dataset
''' + _PEMS_CSV_CLASS + '''

data_dict = {
    'ETTh1': Dataset_ETT_hour,
    'ETTh2': Dataset_ETT_hour,
    'ETTm1': Dataset_ETT_minute,
    'ETTm2': Dataset_ETT_minute,
    'PEMS': Dataset_PEMS_CSV,
    'custom': Dataset_Custom,
}


class _StrideWrap(Dataset):
    def __init__(self, base, stride: int):
        self.base = base
        self.stride = max(1, int(stride))
        self.indices = list(range(0, len(base), self.stride))

    def __getattr__(self, name):
        return getattr(self.base, name)

    def __getitem__(self, index):
        return self.base[self.indices[index]]

    def __len__(self):
        return len(self.indices)


class _WindowCapWrap(Dataset):
    def __init__(self, base, max_windows: int, seed: int):
        import random
        self.base = base
        n = len(base)
        k = int(max_windows)
        if k < 1:
            raise ValueError('max_windows must be >= 1, got %r' % (max_windows,))
        if k >= n:
            self.indices = list(range(n))
        else:
            rng = random.Random(int(seed))
            self.indices = sorted(rng.sample(range(n), k))

    def __getattr__(self, name):
        return getattr(self.base, name)

    def __getitem__(self, index):
        return self.base[self.indices[index]]

    def __len__(self):
        return len(self.indices)


class _IndexWrap(Dataset):
    def __init__(self, base, indices):
        self.base = base
        self.indices = list(indices)
        n = len(base)
        if not self.indices:
            raise ValueError('index wrap is empty')
        if any(i < 0 or i >= n for i in self.indices):
            raise ValueError('index wrap out of range for n=%d' % n)

    def __getattr__(self, name):
        return getattr(self.base, name)

    def __getitem__(self, index):
        return self.base[self.indices[index]]

    def __len__(self):
        return len(self.indices)


def _apply_window_cap(data_set, args, flag):
    if flag == 'pred':
        return data_set
    seed = int(getattr(args, 'window_subset_seed', 42))
    if flag == 'train':
        cap = int(getattr(args, 'train_max_windows', 0) or 0)
        seed = seed + 17
    elif flag == 'val':
        cap = int(getattr(args, 'val_max_windows', 0) or 0)
        seed = seed + 29
    else:
        cap = int(getattr(args, 'test_max_windows', 0) or 0)
        frac = float(getattr(args, 'eval_test_fraction', 0) or 0)
        ref_pred = int(getattr(args, 'eval_ref_pred_len', 0) or 0)
        if cap > 0 and 0.0 < frac < 1.0:
            raise ValueError('test_max_windows and eval_test_fraction cannot both apply')
        if cap < 1 and 0.0 < frac < 1.0 and ref_pred > 0:
            import random
            pred_len = int(args.pred_len)
            stride = 1
            unstrided = data_set
            if isinstance(data_set, _StrideWrap):
                stride = int(data_set.stride)
                unstrided = data_set.base
            n_unstrided_ref = len(unstrided) + pred_len - ref_pred
            if n_unstrided_ref < 1:
                raise ValueError('eval_ref_pred_len=%s leaves no campaign-H windows' % ref_pred)
            n_ref = len(range(0, n_unstrided_ref, stride))
            k = max(1, int(round(n_ref * frac)))
            if k >= n_ref:
                idxs = list(range(n_ref))
            else:
                rng = random.Random(int(seed))
                idxs = sorted(rng.sample(range(n_ref), k))
            n = len(data_set)
            idxs = [i for i in idxs if i < n]
            if not idxs:
                raise ValueError('binary_10pct empty after clip to pred_len=%s' % pred_len)
            return _IndexWrap(data_set, idxs)
        if cap < 1 and 0.0 < frac < 1.0:
            n = len(data_set)
            cap = max(1, int(round(n * frac)))
    if cap > 0:
        data_set = _WindowCapWrap(data_set, cap, seed)
    return data_set


def data_provider(args, flag):
    Data = data_dict[args.data]
    timeenc = 0 if args.embed != 'timeF' else 1

    if flag == 'test':
        shuffle_flag = False
        drop_last = False
        # batch_size=1 so all windows are evaluated (drop_last=False + large
        # batch breaks PatchTST's np.array(preds) on a ragged final batch).
        batch_size = 1
        freq = args.freq
    elif flag == 'pred':
        shuffle_flag = False
        drop_last = False
        batch_size = 1
        freq = args.freq
        Data = Dataset_Pred
    else:
        shuffle_flag = True
        drop_last = True
        batch_size = args.batch_size
        freq = args.freq

    data_set = Data(
        root_path=args.root_path,
        data_path=args.data_path,
        flag=flag,
        size=[args.seq_len, args.label_len, args.pred_len],
        features=args.features,
        target=args.target,
        timeenc=timeenc,
        freq=freq
    )
    if flag == 'train':
        stride = int(getattr(args, 'train_window_stride', 1) or 1)
    elif flag == 'val':
        stride = int(getattr(args, 'val_window_stride', 1) or 1)
    else:
        stride = int(getattr(args, 'test_window_stride', 1) or 1)
    if stride > 1 and flag != 'pred':
        data_set = _StrideWrap(data_set, stride)
    data_set = _apply_window_cap(data_set, args, flag)
    print(flag, len(data_set), f'stride={stride}')
    data_loader = DataLoader(
        data_set,
        batch_size=batch_size,
        shuffle=shuffle_flag,
        num_workers=args.num_workers,
        drop_last=drop_last)
    return data_set, data_loader
'''


def _ensure_cli(path: Path, anchor: str) -> None:
    text = path.read_text(encoding="utf-8")
    if "train_window_stride" in text:
        lines = []
        for line in text.splitlines(True):
            if "train_window_stride" in line or "val_window_stride" in line or "test_window_stride" in line:
                lines.append("    " + line.lstrip())
            else:
                lines.append(line)
        text = "".join(lines)
    elif anchor not in text:
        raise RuntimeError(f"anchor missing in {path}: {anchor!r}")
    else:
        insert = (
            "    parser.add_argument('--train_window_stride', type=int, default=1)\n"
            "    parser.add_argument('--val_window_stride', type=int, default=1)\n"
            "    parser.add_argument('--test_window_stride', type=int, default=1)\n"
            "    " + anchor
        )
        text = text.replace(anchor, insert, 1)
    if "train_max_windows" not in text:
        stride_anchor = "parser.add_argument('--test_window_stride', type=int, default=1)"
        if stride_anchor not in text:
            raise RuntimeError(f"cannot insert window-cap CLI in {path}")
        cap_insert = (
            stride_anchor
            + "\n    parser.add_argument('--train_max_windows', type=int, default=0)\n"
            "    parser.add_argument('--val_max_windows', type=int, default=0)\n"
            "    parser.add_argument('--test_max_windows', type=int, default=0)\n"
            "    parser.add_argument('--eval_test_fraction', type=float, default=0.0)\n"
            "    parser.add_argument('--eval_ref_pred_len', type=int, default=0)\n"
            "    parser.add_argument('--window_subset_seed', type=int, default=42)"
        )
        text = text.replace(stride_anchor, cap_insert, 1)
    elif "eval_test_fraction" not in text:
        cap_anchor = "parser.add_argument('--test_max_windows', type=int, default=0)"
        if cap_anchor not in text:
            raise RuntimeError(f"cannot insert eval_test_fraction CLI in {path}")
        text = text.replace(
            cap_anchor,
            cap_anchor
            + "\n    parser.add_argument('--eval_test_fraction', type=float, default=0.0)"
            + "\n    parser.add_argument('--eval_ref_pred_len', type=int, default=0)",
            1,
        )
    elif "eval_ref_pred_len" not in text:
        frac_anchor = "parser.add_argument('--eval_test_fraction', type=float, default=0.0)"
        if frac_anchor not in text:
            raise RuntimeError(f"cannot insert eval_ref_pred_len CLI in {path}")
        text = text.replace(
            frac_anchor,
            frac_anchor
            + "\n    parser.add_argument('--eval_ref_pred_len', type=int, default=0)",
            1,
        )
    path.write_text(text, encoding="utf-8")


def _patch_test_checkpoint_path(path: Path) -> None:
    """is_training=0 was loading ./checkpoints/ instead of --checkpoints."""
    if not path.is_file():
        raise FileNotFoundError(path)
    text = path.read_text(encoding="utf-8")
    needle = "os.path.join('./checkpoints/' + setting, 'checkpoint.pth')"
    repl = "os.path.join(self.args.checkpoints, setting, 'checkpoint.pth')"
    if needle not in text:
        if repl in text:
            return
        raise RuntimeError(f"test() checkpoint join missing in {path}")
    path.write_text(text.replace(needle, repl), encoding="utf-8")


def _stub_reformer_import() -> None:
    """iTransformer hard-imports reformer_pytorch even when model=iTransformer."""
    if not ITRANS_ATTN.is_file():
        return
    text = ITRANS_ATTN.read_text(encoding="utf-8")
    if "CANVAS128_REFORMER_STUB" in text:
        return
    needle = "from reformer_pytorch import LSHSelfAttention"
    if needle not in text:
        raise RuntimeError(f"reformer import missing in {ITRANS_ATTN}")
    stub = (
        "# CANVAS128_REFORMER_STUB\n"
        "try:\n"
        "    from reformer_pytorch import LSHSelfAttention\n"
        "except ImportError:\n"
        "    class LSHSelfAttention(nn.Module):\n"
        "        def __init__(self, *args, **kwargs):\n"
        "            super().__init__()\n"
        "            raise ImportError('reformer_pytorch required only for Reformer models')\n"
    )
    ITRANS_ATTN.write_text(text.replace(needle, stub, 1), encoding="utf-8")


def assert_stride_wrap_present() -> None:
    for path in (ITRANS_FACTORY, PATCH_FACTORY):
        if not path.is_file():
            raise FileNotFoundError(path)
        text = path.read_text(encoding="utf-8")
        if "_StrideWrap" not in text:
            raise RuntimeError(f"_StrideWrap missing in {path}; patches not applied")
        if "Dataset_PEMS_CSV" not in text:
            raise RuntimeError(f"Dataset_PEMS_CSV missing in {path}; patches not applied")


def main() -> int:
    if not ITRANS_FACTORY.is_file() or not PATCH_FACTORY.is_file():
        raise FileNotFoundError("clone temp/iTransformer and temp/PatchTST first")
    ITRANS_FACTORY.write_text(FACTORY_BODY_ITRANS, encoding="utf-8")
    _ensure_cli(ITRANS_RUN, "parser.add_argument('--partial_start_index'")
    print(f"[ok] patched {ITRANS_FACTORY}")
    PATCH_FACTORY.write_text(FACTORY_BODY_PATCH, encoding="utf-8")
    _ensure_cli(PATCH_RUN, "parser.add_argument('--random_seed'")
    print(f"[ok] patched {PATCH_FACTORY}")
    _stub_reformer_import()
    print(f"[ok] reformer stub in {ITRANS_ATTN}")
    _patch_test_checkpoint_path(ITRANS_EXP)
    print(f"[ok] test ckpt path {ITRANS_EXP}")
    _patch_test_checkpoint_path(PATCH_EXP)
    print(f"[ok] test ckpt path {PATCH_EXP}")
    assert_stride_wrap_present()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
