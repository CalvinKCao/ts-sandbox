"""Quick test for dataset loading and data leakage verification."""

import sys
import os

script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, script_dir)

from test_etth1_comparison import ETTh1Dataset, ETTH1_PATH, LOOKBACK_LENGTH, FORECAST_LENGTH

def main():
    # Test dataset loading
    print('Loading train split...')
    train_ds = ETTh1Dataset(ETTH1_PATH, LOOKBACK_LENGTH, FORECAST_LENGTH, split='train')
    print(f'Train samples: {len(train_ds)}')

    print('Loading val split...')
    val_ds = ETTh1Dataset(ETTH1_PATH, LOOKBACK_LENGTH, FORECAST_LENGTH, split='val')
    print(f'Val samples: {len(val_ds)}')

    print('Loading test split...')
    test_ds = ETTh1Dataset(ETTH1_PATH, LOOKBACK_LENGTH, FORECAST_LENGTH, split='test')
    print(f'Test samples: {len(test_ds)}')

    # Test a single sample
    sample = train_ds[0]
    print(f'\nSample shapes:')
    print(f'  past_ts: {sample["past_ts"].shape}')
    print(f'  past_img: {sample["past_img"].shape}')
    print(f'  future_ts: {sample["future_ts"].shape}')
    print(f'  mean: {sample["mean"].shape}')
    print(f'  std: {sample["std"].shape}')

    # Verify no overlap between splits (index overlap, not window overlap)
    train_indices = set(train_ds.indices)
    val_indices = set(val_ds.indices)
    test_indices = set(test_ds.indices)

    overlap_train_val = train_indices.intersection(val_indices)
    overlap_val_test = val_indices.intersection(test_indices)
    overlap_train_test = train_indices.intersection(test_indices)

    print(f'\nSplit overlap check (index overlap):')
    print(f'  Train-Val overlap: {len(overlap_train_val)} (should be 0)')
    print(f'  Val-Test overlap: {len(overlap_val_test)} (should be 0)')
    print(f'  Train-Test overlap: {len(overlap_train_test)} (should be 0)')

    if overlap_train_val or overlap_val_test or overlap_train_test:
        print('WARNING: Data leakage detected!')
        return False
    
    # Check that splits are chronological
    max_train = max(train_ds.indices)
    min_val = min(val_ds.indices)
    max_val = max(val_ds.indices)
    min_test = min(test_ds.indices)
    
    print(f'\nChronological order check:')
    print(f'  Train max index: {max_train}')
    print(f'  Val min index: {min_val}')
    print(f'  Val max index: {max_val}')
    print(f'  Test min index: {min_test}')
    
    if max_train < min_val and max_val < min_test:
        print('  ✓ Splits are properly chronological!')
    else:
        print('  WARNING: Splits may not be properly chronological!')
        return False

    print('\n✓ Dataset loading test passed!')
    return True

if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)

