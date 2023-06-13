# Changelog

## Version 0.1.5

[@cumbof/hdlib@0.1.5](https://github.com/cumbof/hdlib/releases/tag/0.1.5)

### New features

- Add `bind`, `bundle`, and `permute` as `Vector` class methods for applying arithmetic operations inplace;
- Rename `split_dataset` in the `parser` module into `kfolds_split`;
- Add `percentage_split` function to the `parser` module;
- Integrate [chopin2](https://github.com/cumbof/chopin2) ML model into the `model` module withe the `Model` class.

## Version 0.1.4

[@cumbof/hdlib@0.1.4](https://github.com/cumbof/hdlib/releases/tag/0.1.4)

### New features

- Add `parser` module with utility functions for dealing with input datasets;
- Check if the input pickle file exists before initializing `Vector` and `Space` objects with `from_file`;
- Report `Vector` and `Space`objects information when calling `print`;
- Add `examples/chopin2.py`: reimplementation of the [chopin2](https://github.com/cumbof/chopin2) ML model with `hdlib`.

### Fixes

- Tags are maintained as sets when applying arithmetic operators.

## Version 0.1.3

[@cumbof/hdlib@0.1.3](https://github.com/cumbof/hdlib/releases/tag/0.1.3)

### New features

- Check for version compatibility when loading a pickle file;
- Add `normalize()` function to the `Vector` class;
- Link vectors in space with `parents` and `children` sets as class attributes;
- Add `link()` and `set_root()` functions to the `Space` class.

## Version 0.1.2

[@cumbof/hdlib@0.1.2](https://github.com/cumbof/hdlib/releases/tag/0.1.2)

### New features

- New distance metrics: `cosine`, `hamming`, and `euclidean`;
- Tags are inherited after applying the arithmetic operators;
- Unit tests for `Vector` and `Space` classes and for `bundle`, `bind`, and `permute` arithmetic operators;
- `What is the Dollar of Mexico?` as a unit test.

## Version 0.1.1

[@cumbof/hdlib@0.1.1](https://github.com/cumbof/hdlib/releases/tag/0.1.1)

### New features

- Vectors can be tagged;
- Tags can be used to retrieve groups of Vectors in the Space with the `get` method of the `Space` class;
- The performances of the `remove` and `findAll` methods of the `Space` class have been improved.

## Version 0.1.0

[@cumbof/hdlib@0.1.0](https://github.com/cumbof/hdlib/releases/tag/0.1.0)

First public release of `hdlib`.

Hyperdimensional Computing Library for building Vector Symbolic Architectures in Python 3
