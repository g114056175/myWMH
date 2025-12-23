## ADDED Requirements

### Requirement: Lazy Data Filtering
The dataset loader SHALL defer empty slice filtering until actual data access to avoid loading entire dataset into memory during initialization.

#### Scenario: Dataset initialization with lazy filtering
- **WHEN** WMHDataset25D is instantiated with a 10GB training directory
- **THEN** initialization completes in less than 30 seconds
- **AND** memory usage during initialization is less than 500MB

#### Scenario: Runtime filtering on access
- **WHEN** a batch is requested during training
- **THEN** empty slice filtering occurs dynamically per volume
- **AND** filtered slices are cached for subsequent access

### Requirement: Timeout Protection
The dataset loader SHALL implement configurable timeouts for all file I/O operations to prevent indefinite hangs.

#### Scenario: File load timeout
- **WHEN** a NIfTI file takes longer than the configured timeout (default 30s)
- **THEN** the operation raises a TimeoutError
- **AND** the error is logged with file path information

#### Scenario: Corrupted file handling
- **WHEN** a corrupted NIfTI file is encountered
- **THEN** the file is skipped with a warning log
- **AND** training continues with remaining valid samples

### Requirement: Progress Monitoring
The dataset loader SHALL provide visible progress feedback during long-running operations.

#### Scenario: Dataset collection progress
- **WHEN** collecting samples from training directory
- **THEN** a tqdm progress bar displays current case being processed
- **AND** total number of cases and slices found

#### Scenario: First epoch loading feedback
- **WHEN** the first training epoch begins
- **THEN** data loading time per batch is logged
- **AND** average loading time is reported after epoch

### Requirement: Memory-Efficient Loading
The dataset loader SHALL use memory-mapped file access where possible to reduce RAM usage.

#### Scenario: Large volume access
- **WHEN** accessing a large NIfTI volume (\u003e1GB)
- **THEN** the loader uses memory mapping instead of full load
- **AND** only requested slices are loaded into memory

### Requirement: Resource-Aware Worker Configuration
The DataLoader SHALL automatically configure num_workers based on available system resources and dataset characteristics.

#### Scenario: Large dataset worker tuning
- **WHEN** creating DataLoader with dataset containing \u003e1000 samples
- **THEN** num_workers is set to min(4, cpu_count - 1) if not specified
- **AND** worker count is reduced to 0 if memory-constrained

#### Scenario: Small dataset optimization
- **WHEN** creating DataLoader with dataset containing \u003c100 samples
- **THEN** num_workers is set to 0 to avoid overhead
