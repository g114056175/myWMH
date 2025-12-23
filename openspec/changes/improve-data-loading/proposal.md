# Change: Improve Data Loading Robustness and Performance

## Why

Current data loading implementation has critical issues when handling large medical imaging datasets (~10GB):
1. **Memory overload**: `_filter_empty_slices()` loads all NIfTI files into memory during dataset initialization
2. **No timeout handling**: Extended load times cause application to hang without feedback
3. **No error recovery**: Crashes occur without graceful degradation
4. **Poor progress visibility**: No monitoring during long-running data operations

These issues prevent successful model training on the WMH dataset.

## What Changes

- **Lazy filtering**: Remove eager loading in `_filter_empty_slices()`, defer to actual data access
- **Timeout protection**: Add configurable timeouts for file I/O operations
- **Progress monitoring**: Add tqdm progress bars and logging for data loading stages
- **Error handling**: Implement try-catch with fallbacks for corrupted files
- **Memory optimization**: Use memory-mapped loading where possible
- **Worker tuning**: Auto-detect safe num_workers based on dataset size

## Impact

- Affected specs: `wmh-data-pipeline` (new capability spec)
- Affected code: 
  - `dataset.py`: WMHDataset25D class refactor
  - `train.py`: Add data loading validation step
- **BREAKING**: Changes `WMHDataset25D` initialization behavior (now lazy)
