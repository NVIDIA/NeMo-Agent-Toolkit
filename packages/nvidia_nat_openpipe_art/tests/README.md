# OpenPipe ART Plugin Tests

This directory contains comprehensive unit tests for the OpenPipe ART plugin components.

## Test Structure

- `test_trainer.py` - Tests for the `ARTTrainer` class
- `test_trainer_adapter.py` - Tests for the `ARTTrainerAdapter` class
- `test_trajectory_builder.py` - Tests for the `ARTTrajectoryBuilder` class
- `conftest.py` - Shared fixtures and configuration

## Running Tests

### Run all tests in this package:
```bash
pytest packages/nvidia_nat_openpipe_art/tests/ -v
```

### Run specific test files:
```bash
pytest packages/nvidia_nat_openpipe_art/tests/test_trainer.py -v
pytest packages/nvidia_nat_openpipe_art/tests/test_trainer_adapter.py -v
pytest packages/nvidia_nat_openpipe_art/tests/test_trajectory_builder.py -v
```

### Run with coverage:
```bash
pytest packages/nvidia_nat_openpipe_art/tests/ --cov=nat.plugins.openpipe --cov-report=term-missing
```

### Run specific test classes or methods:
```bash
# Run a specific test class
pytest packages/nvidia_nat_openpipe_art/tests/test_trainer.py::TestARTTrainer -v

# Run a specific test method
pytest packages/nvidia_nat_openpipe_art/tests/test_trainer.py::TestARTTrainer::test_trainer_initialization -v
```

## Test Coverage

The tests provide comprehensive coverage for:

### ARTTrainer (`test_trainer.py`)
- Initialization and configuration
- Running single and multiple epochs
- Handling trajectories and empty collections
- Validation evaluation
- Curriculum learning (enabled/disabled, expansion, filtering)
- Progress logging and visualization
- Resource cleanup
- Error handling and recovery

### ARTTrainerAdapter (`test_trainer_adapter.py`)
- Initialization and health checks
- Episode validation and ordering
- Trajectory group construction
- Job submission and management
- Status tracking (running, completed, failed, cancelled)
- Checkpoint management
- Progress logging
- Error handling

### ARTTrajectoryBuilder (`test_trajectory_builder.py`)
- Initialization and configuration
- Starting and managing evaluation runs
- Finalizing and building trajectories
- Grouping trajectories by example ID
- Handling various message types (user, assistant, tool, etc.)
- Episode validation and filtering
- Error handling and parse failures
- Progress logging
