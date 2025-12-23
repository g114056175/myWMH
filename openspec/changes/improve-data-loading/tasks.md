## 1. Data Loading Core Improvements

- [ ] 1.1 Refactor `_filter_empty_slices()` to use lazy evaluation
- [ ] 1.2 Add timeout wrapper for NIfTI file loading operations
- [ ] 1.3 Implement progress monitoring with tqdm
- [ ] 1.4 Add robust error handling with logging

## 2. Memory Optimization

- [ ] 2.1 Implement memory-mapped loading for large volumes
- [ ] 2.2 Add validation step before full dataset initialization
- [ ] 2.3 Optimize num_workers selection based on system resources

## 3. Training Script Updates

- [ ] 3.1 Add pre-training data validation check
- [ ] 3.2 Implement checkpoint resume from interrupted training
- [ ] 3.3 Add memory usage monitoring during training

## 4. Testing and Validation

- [ ] 4.1 Test data loading with timeout scenarios
- [ ] 4.2 Verify memory usage stays within limits
- [ ] 4.3 Run full training pipeline on data/wmh/training
- [ ] 4.4 Validate on data/wmh/test
