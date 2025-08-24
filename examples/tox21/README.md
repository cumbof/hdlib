# Toxicology in the 21st Century – Vector-Symbolic Architectures for the Classification of Chemical Compounds

This example demonstrates how to use `hdlib` to create vector-symbolic representations of chemical compound structures from the [Tox21 dataset](https://github.com/cumbof/hdlib/blob/main/examples/tox21/tox21.csv). The script in this folder shows how to convert SMILES strings into high-simensional vectors suitable for machine learning tasks.

## Encoding Techniques

This example showcases three distinct methods for encoding the SMILES structures:
- __Atom-wise Encoding__: generates a hypervector for a comound by bundling the hypervectors of its individual atoms;
- __K-mer-based Encoding__: treats the SMILES string as a sequence and creates hypervectors for overlapping k-mers (substrings of length _k_);
- __Fragment-based Encoding__: decomposes the chemical structure into funcitonal groups or fragments and bundles their corresponding hypervectors.
