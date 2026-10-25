# Machine Translation with PyTorch

An ongoing personal project to implement a Transformer-based neural machine translation model in PyTorch, translating between English and Spanish.

Rather than relying entirely on a prebuilt Transformer implementation, this project explores the internal components of the architecture, including attention mechanisms, encoder and decoder layers, embeddings, and positional encoding. It also includes a custom tokenizer based on character-level Byte Pair Encoding (BPE).

**Project status:** Active development. The model implementation, training pipeline, and translation capabilities are still being developed and evaluated.

## Project Goals

- Implement a Transformer encoder-decoder architecture using PyTorch.
- Develop custom implementations of the core Transformer components.
- Build a BPE tokenizer for processing English and Spanish text.
- Train a neural machine translation model on a parallel text corpus.
- Develop an autoregressive inference pipeline for generating translations.
- Explore the relationship between tokenization, model architecture, and translation quality.

## Implementation

### Transformer Architecture

The model is being implemented using PyTorch, with custom implementations of its core components rather than relying on a complete, prebuilt Transformer model.

The implementation explores:

- Token embeddings.
- Multi-head self-attention.
- Encoder and decoder layers.
- Cross-attention between encoder and decoder representations.
- Feed-forward networks and output projections.
- Autoregressive decoding for translation generation.

### Custom BPE Tokenizer

The project includes a custom tokenizer based on character-level Byte Pair Encoding.

The tokenizer learns merge rules from a text corpus and applies them to segment words into subword tokens. The resulting tokens are mapped to integer IDs for use as model inputs and outputs.

The tokenizer implementation includes:

- Text normalization and preprocessing.
- Initial character-level vocabulary construction.
- Iterative BPE merge-rule learning.
- Vocabulary generation and token-to-ID mapping.
- Encoding and decoding text.

### Training and Inference

The intended pipeline uses parallel English-Spanish sentences to train the encoder-decoder Transformer. During inference, the model generates a translation token by token, using previously generated tokens to predict subsequent ones.

From preliminary tests, the model seems to able to pick up certain aspects of spanish grammar, however this hasn't been quantified. Addtionally, due to translation pairs with the same english sentence but different spanish translations, the model's training loss is unable to drop to a level required for proper translation. 

To address this, I am switching the dataset I am using to a cleaner one from hugging face. I am also moving from local training to cloud training to allow for faster iteration.

The training and inference workflows are still under development, and the repository should not yet be considered a complete translation system.

## Repository Structure

```text
machine_translation/
├── tokenizers/              # Saved tokenizer dictionaries
├── custom_tokenizer.py      # Custom BPE tokenizer implementation
├── machine_translation.ipynb # Transformer implementation and experiments
├── tokenizer_test.ipynb     # Tokenizer testing and experiments
├── README.md
└── LICENSE
```

## Development Environment

- **Python**
- **PyTorch**
- **NumPy**
- **Jupyter Notebook**

The exact dependencies may change as development progresses.

## Current Status and Limitations

This project is actively being developed, and its implementation is subject to change.

the current transformer has been fully implemented along with a corresponding BPE tokenizer. Work is being done to train and validate the implementation and document translation results.

Current limitations include:

- The model and training pipeline are not yet fully stabilized.
- Translation quality has not been systematically evaluated.
- No validated benchmark results are currently reported.
- The code organization and interfaces may change as components are refined.

Because reliable evaluation results are not yet available, this README does not claim a particular translation accuracy or performance level.

## Future Work

- Refine and modularize the Transformer implementation.
- Research different datasets with larger corpuses and better translation examples
- Improve the efficiency of BPE training and corpus tokenization.
- Develop a reliable autoregressive inference workflow.
- Evaluate translation quality using held-out data and appropriate metrics, such as BLEU.
- Compare generated translations against reference translations and document representative examples.

## License

See [LICENSE](LICENSE) for licensing information.

