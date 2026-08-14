# Intent Classifier - Standalone Module

## Overview

The `intent_classifier` module is a **standalone/branched application** that is **not integrated** into the main chat or agent flow of the Mini AI Assistant. It exists as a separate ML module for potential future integration or standalone use.

## Purpose

This module demonstrates a custom transformer-based text classification implementation with:

- Custom tokenizer and vocabulary management
- Transformer encoder with multi-head attention
- Positional encoding
- Mean pooling and classification head
- Training pipeline with early stopping and checkpointing
- Inference wrapper for predictions

## Architecture

```
Input Text → Tokenizer → Embedding → Positional Encoding
    → Transformer Encoder (Multi-Head Attention)
    → Mean Pooling → Classifier → Intent Label
```

## Key Components

### Model (`model.py`)
- `IntentClassifierModel`: Main model combining all components
- Configurable: vocab_size, d_model, nhead, num_layers, num_classes

### Encoder (`encoder.py`)
- `TransformerEncoder`: Stack of encoder layers
- Uses `MultiHeadAttention` and `PositionalEncoding`

### Attention (`attention.py`)
- `MultiHeadAttention`: Multi-head self-attention mechanism
- Supports query, key, value projections

### Tokenizer (`tokenizer.py`)
- Custom tokenizer with vocabulary management
- Text encoding and decoding
- Special token handling

### Training (`trainer.py`)
- Full training loop with validation
- Early stopping support
- Checkpoint saving/loading
- Metrics tracking (loss, accuracy)

### Inference (`predictor.py`)
- `Predictor`: Wrapper for model inference
- `ClassificationResult`: Result dataclass with label and confidence

### Supporting Components
- `config.py`: Model configuration
- `labels.py`: Intent label definitions
- `dataset.py`: Dataset handling and loading
- `checkpoint.py`: Model checkpoint management
- `early_stopping.py`: Early stopping logic
- `metrics.py`: Training metrics calculation
- `pooling.py`: Pooling strategies (mean, max)
- `positional_encoding.py`: Sinusoidal positional encoding

## Usage

### Training
```bash
python train.py
```

### Prediction
```bash
python predict.py
```

### As a Module
```python
from intent_classifier.predictor import Predictor
from intent_classifier.config import ModelConfig

config = ModelConfig()
predictor = Predictor(config, checkpoint_path="checkpoints/best.pt")
result = predictor.predict("Find me a laptop")
print(result.label, result.confidence)
```

## Data Format

Training data is in JSONL format:
```jsonl
{"text": "Find me a laptop", "label": "search_product"}
{"text": "What is the return policy?", "label": "ask_knowledge"}
{"text": "Hello, how are you?", "label": "chat"}
```

## Model Checkpoints

Trained models are saved in `checkpoints/intent/`:
- `best.pt`: Best validation checkpoint
- `last.pt`: Last training checkpoint

## Integration Status

**Current Status:** Not integrated into main application

**Potential Future Integration:**
- Could be used for intent-based routing in `AssistantEngine`
- Could provide confidence scores for tool selection
- Could be integrated via the `Router` class in `application/router/`

## Dependencies

- PyTorch
- NumPy
- tqdm

## Notes

This module is kept separate from the main application to:
1. Demonstrate clean separation of concerns
2. Allow independent development and testing
3. Enable potential future integration without affecting core functionality
4. Serve as a reference for custom transformer implementations
