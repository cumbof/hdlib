# Quantum Hyperdimensional Computing

This directory contains examples of how to use the `hdlib` library to implement Quantum Hyperdimensional Computing (QHDC) models.

## Overview

**Quantum Hyperdimensional Computing (QHDC)** is a neuromorphic computing paradigm that adapts the principles of classical Hyperdimensional Computing (HDC) for quantum hardware. It provides a direct, resource-efficient mapping between the core HDC algebra (bundling, binding, permutation) and native quantum algorithms (Linear Combination of Unitaries, Phase Oracles, and the Quantum Fourier Transform).

These examples demonstrate how to build and run end-to-end QHDC workflows.

## Examples

### Reasoning

This example demonstrates how QHDC can be used to solve analogical reasoning problems (e.g., "A is to B as C is to ?").

- __What it does:__ The script encodes symbolic information into quantum states (hypervectors). It then uses quantum operations to perform the reasoning task by finding the vector D that best completes the analogy A:B::C:D.

- __Key QHDC Operations:__
  - Encoding: Classical symbols are mapped to phase-encoded quantum states.
  - Binding/Permutation: The relationships between symbols are constructed.
  - Similarity Search: The Hadamard Test is used to find the best match from a set of candidates.

### Supervised Learning

This example implements a full, data-driven supervised classification pipeline using QHDC. It demonstrates how to train a model and perform inference on a dataset (e.g., simplified MNIST digits).

- __What it does:__ The script trains a QHDC model by building "prototype" hypervectors for each class. It then classifies new, unseen data points by finding the prototype with the highest quantum-measured similarity.

- __Key QHDC Operations:__
  - Encoding: Data features are encoded into quantum states.
  - Bundling (Training): The Linear Combination of Unitaries (LCU) algorithm is used to average all training vectors into a single "prototype" quantum state for each class.
  - Inference: The Hadamard Test is used to measure the similarity between a new data point and the class prototypes, enabling classification.