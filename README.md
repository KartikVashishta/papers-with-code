# Deep Learning Paper Implementations

This repository serves as a personal learning journey through important papers in deep learning, starting with foundational architectures and gradually expanding to more complex models. Each implementation is meant to be a clean, educational reference point with a focus on understanding the core concepts.

## Current Implementations

| Paper                                                                                                                                                        | Implementation                                             | Key Concepts                                                                                                            |
| ------------------------------------------------------------------------------------------------------------------------------------------------------------ | ---------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------- |
| [Attention Is All You Need](https://arxiv.org/abs/1706.03762)                                                                                                | [transformer-implementation/](transformer-implementation/) | - Multi-Head Attention<br>- Positional Encoding<br>- Layer Normalization<br>- Label Smoothing<br>- Warmup Learning Rate |
| [Neural Machine Translation by Jointly Learning to Align and Translate](https://arxiv.org/abs/1409.0473)                                                     | [BPE/](BPE/)                                               | - Byte Pair Encoding<br>- Subword Tokenization<br>- Vocabulary Building<br>- Special Token Handling                     |
| [Language Models are Unsupervised Multitask Learners](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf) | [gpt-2/](gpt-2/)                                           | - Transformer Decoder<br>- Autoregressive Language Modeling<br>- Transfer Learning<br>- Advanced Text Generation        |

## Transformer Implementation Details

The current implementation includes a complete transformer architecture with:

- Multi-headed self-attention mechanism
- Position-wise feed-forward networks
- Positional encodings
- Layer normalization
- Encoder and decoder stacks
- Label smoothing
- Learning rate scheduling with warmup

## BPE Tokenizer Details

The BPE (Byte Pair Encoding) tokenizer implementation is inspired by Sebastian Raschka's work and includes:

- Complete training algorithm to learn subword tokens from a corpus
- Efficient encoding and decoding methods with merge prioritization
- Full support for special tokens and Unicode characters
- Space preprocessing using 'Ġ' character (following GPT tokenizer convention)
- OpenAI-compatible format loader for GPT-2 vocabularies
- Performance optimizations with caching mechanisms
- Regex-based tokenization for faster processing

## GPT-2 Implementation Details

The GPT-2 implementation is inspired by Andrej Karpathy's work with many optimizations to be made. It features:

- Transformer decoder architecture
- Autoregressive language modeling
- Pre-training and fine-tuning capabilities
- Text generation with various sampling strategies (temperature, top-k, top-p)
- Efficient attention patterns for improved training
- Educational implementation focusing on clarity and understanding

## Note

These implementations are meant for educational purposes and self-reference. While they aim to be correct, they may not be optimized for production use. They serve as a starting point for understanding the underlying concepts and architectures described in the papers.
