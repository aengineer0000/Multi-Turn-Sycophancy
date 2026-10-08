# Project: Multi-Turn Sycophancy Probing (CS Education)

## Core Guidelines
1. Tensor Documentation: Whenever manipulating PyTorch tensors or extracting activations via TransformerLens, explicitly comment tensor dimensions at every step (e.g., `# [batch_size, seq_len, d_model]`).
2. Determinism & Seeds: Every script running inference, data generation, or Scikit-Learn training must accept a `--seed` argument and set random states globally.
3. Separation of Concerns: Keep dataset generation (`src/data/`), model caching (`src/extraction/`), probing (`src/probing/`), and plotting (`src/visualization/`) in distinct modules.
4. Logging: Do not use bare `print()` statements. Use standard Python `logging` to log layer extraction progress, batch sizes, and classification metrics.
5. N-Turn Extensibility: Never hardcode turn indices (e.g., `turn_1`, `turn_2`) into core classes. Represent dialogues as dynamic lists of `Turn` objects so the pipeline can support 3-turn, 5-turn, or N-turn dialogues seamlessly, even if the current default configuration is 3 turns.