
# Custom Trainer Package

This package includes custom implementations for training machine learning models with early stopping and debugging capabilities. It comprises the following main components:

1. `CustomSFTTrainer`
2. `DebugSFTTrainer`
3. `CustomEarlyStoppingCallback`

## Components

### 1. `CustomSFTTrainer`

The `CustomSFTTrainer` class is designed for supervised fine-tuning (SFT) of machine learning models. It includes various methods to handle the training process efficiently.

Key methods and attributes:
- `__init__(tokenizer, embedding_model, *args, **kwargs)`: Initialises the trainer with specific configurations.
- `create_optimizer()`: Creates the optimizer.
- `get_sentence_embedding(sentence)`: Gets the embedding of a sentence.
- `compute_loss(model, inputs, return_outputs=False)`: Computes the loss for training.
- `training_step(model, inputs)`: Performs a training step on a batch of inputs.
- `train()`: Manages the overall training loop, including evaluation and gradient accumulation.

### 2. `DebugSFTTrainer`

The `DebugSFTTrainer` class extends the `CustomSFTTrainer` with additional debugging functionalities. This includes detailed logging and monitoring of the training process.

Key methods and attributes:
- `__init__(*args, **kwargs)`: Initialises the debug trainer with specific configurations.
- `create_optimizer()`: Creates the optimizer.
- `compute_loss(model, inputs, return_outputs=False)`: Computes the loss for training with debugging information.
- `training_step(model, inputs)`: Performs a training step on a batch of inputs with debugging information.
- `train()`: Manages the overall training loop with enhanced debugging features.

### 3. `CustomEarlyStoppingCallback`

The `CustomEarlyStoppingCallback` class is a custom implementation of an early stopping callback. It monitors the training process and stops training when no improvement is observed after a certain number of evaluations.

Key methods and attributes:
- `__init__(early_stopping_patience=30, early_stopping_threshold=0.01)`: Initialises the callback with patience and threshold parameters.
- `on_evaluate(args, state, control, metrics=None, **kwargs)`: Checks for improvement during evaluation.
- `_is_improvement(current_metric)`: Checks if the current metric is an improvement.

### `__init__.py`

This file initialises the package by importing the main classes.

Imports:
- `CustomSFTTrainer`
- `DebugSFTTrainer`
- `CustomEarlyStoppingCallback`

## License

This project is licensed under the Apache 2.0 License. See the [LICENSE](https://www.apache.org/licenses/LICENSE-2.0) file for details.