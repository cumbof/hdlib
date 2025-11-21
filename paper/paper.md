---
title: 'hdlib 2.0: extending machine learning capabilities of Vector-Symbolic Architectures'
tags:
  - Vector-Symbolic Architectures
  - Hyperdimensional Computing
  - Machine Learning
  - Quantum Machine Learning
  - Python
  - Library
authors:
  - name: Fabio Cumbo
    orcid: 0000-0003-2920-5838
    corresponding: true
    affiliation: 1
  - name: Kabir Dhillon
    orcid: 0009-0000-3830-1405
    affiliation: 2
  - name: Daniel Blankenberg
    orcid: 0000-0002-6833-9049
    affiliation: "1, 3"
affiliations:
  - name: Center for Computational Life Sciences, Cleveland Clinic Research, Cleveland Clinic, Cleveland, Ohio, United States of America
    index: 1
  - name: College of Engineering, Ohio State University, Columbus, Ohio, United States of America
    index: 2
  - name: Department of Molecular Medicine, Cleveland Clinic Lerner College of Medicine, Case Western Reserve University, Cleveland, Ohio, United States of America
    index: 3
date: 20 November 2025
bibliography: paper.bib
---

# Summary

Following the initial publication of _hdlib_ [@cumbo2023hdlib], a Python library for designing Vector-Symbolic Architectures (VSA), we introduce a major extension that significantly enhances its machine learning capabilities. VSA, also known as Hyperdimensional Computing, is a computing paradigm that represents and processes information using high-dimensional vectors. While the first version of _hdlib_ established a foundation for manipulating these vectors, this update addresses the growing need for more advanced modeling within the VSA framework. Here, we present four extensions: significant enhancements to the existing supervised classification model also enabling feature selection, and new regression, clustering, and graph-based learning models. Furthermore, we propose the first implementation ever of Quantum Hyperdimensional Computing with quantum-powered arithmetic operations and a new Quantum Machine Learning model for supervised learning [@cumbo2025qhdc].

_hdlib_ remains open-source and available on GitHub at [https://github.com/cumbof/hdlib](https://github.com/cumbof/hdlib), and distributed through the Python Package Index (_pip install hdlib_) and Conda (_conda install -c conda-forge hdlib_).

# Statement of need

The successful application of VSA across diverse scientific domains has created a demand for more sophisticated machine learning models that go beyond basic classification. Researchers now require tools to tackle regression tasks, model complex relationships in structured data like graphs, and better optimize models by identifying the most salient features.

This new version of _hdlib_ directly addresses this need. While other libraries provide foundational VSA operations [@simon2022hdtorch; @heddes2023torchhd; @kang2022openhd], _hdlib_ now introduces a cohesive toolkit for advanced machine learning that is, to our knowledge, unique in its integration of regression, clustering, graph encoding, and enhanced feature selection within a single framework. These additions empower researchers to move from rapid prototyping of core VSA concepts to building and evaluating complex machine learning pipelines that are now used in the context of different problems in different scientific domains [@cumbo2025hyperdimensional; @cumbo2025feature; @joshi2025large; @cumbo2020brain; @cumbo2025novel; @cumbo2025predicting; @cumbo2025designing].

# Extending Machine Learning functionalities

The new architecture is summarized in \autoref{fig:overview}.

![An overview of the _hdlib_ 2.0 library architecture, highlighting the distinction between the original (top, transparent) and new components (bottom). Foundational classes from version 1.0 include `hdlib.space.Space` (Class 1), `hdlib.vector.Vector` (Class 2), `hdlib.arithmetic` module (Class 3), and the `hdlib.model.classification.ClassificationModel` (Class 4). This work introduces major new functionalities through the `hdlib.model` module comprising the new `clustering.ClusteringModel` (Class 5), `regression.RegressionEncoder` (Class 6) and `regression.RegressionModel` (Class 7), and `graph.GraphModel` (Class 8), creating a comprehensive toolkit for VSA-based machine learning.\label{fig:overview}](hdlib.pdf)

We have significantly expanded `hdlib.model` to cover a broader spectrum of learning paradigms:

- __Classification__: the `ClassificationModel` now features advanced model optimization tools. We introduced an improved stepwise regression method for identifying feature importance and an `auto_tune` method that performs parameter sweeps to optimize vector dimensionality and level vectors;

- __Clustering__: the new `ClusteringModel` implements k-means clustering in hyperspace [@gupta2022store]. It iteratively refines centroids by bundling constituent data hypervectors, using cosine similarity to assign points to clusters until convergence;

- __Regression__: we implemented a `RegressionModel` and `RegressionEncoder` based on the RegHD algorithm [@hernandez2021reghd]. This module maps inputs to a high-dimensional manifold and employs a multi-model strategy, where final predictions are a confidence-weighted sum of multiple regression models trained on distinct clusters of the data;

- __Graphs__: the `GraphModel` encodes directed and undirected weighted graphs into single hypervectors [@poduval2022graphd]. By compressing node neighborhoods and edge weights into a unified representation, this model enables efficient graph classification and includes an error mitigation routine to refine the graph representation.

# Quantum Hyperdimensional Computing

We introduce Quantum Hyperdimensional Computing (QHDC), a foundational paradigm designed to run on quantum devices [@cumbo2025qhdc]. The `hdlib.arithmetic.quantum` module implements the QHDC arithmetic using IBM's Qiskit framework [@javadiabhari2024quantumcomputingqiskit]:

- __Encoding__: we employ a phase encoding strategy where bipolar hypervectors are mapped to the relative phases of a uniform superposition state, enabling efficient algebraic manipulation;

- __Binding__: realized via quantum phase oracles that map element-wise multiplication to the sequential application of phases;

- __Bundling__: implemented as a quantum-native averaging process using a Linear Combination of Unitaries (LCU) [@chakraborty2024implementing] followed by Oblivious Amplitude Amplification (OAA) [@guerreschi2019repeat];

- __Permutation__: achieved using the Quantum Fourier Transform (QFT) [@weinstein2001implementation] to induce cyclic shifts in the computational basis;

- __Similarity__: computed via the Hadamard Test to estimate the real part of the inner product (cosine similarity) between quantum states.

Finally, we provide a `QuantumClassificationModel` for supervised learning as a totally new approach to Quantum Machine Learning.

With the integration of these modules, `hdlib` 2.0 provides the scientific community with a unified and powerful framework, paving the way for the development of novel, brain-inspired solutions to a broader spectrum of machine learning problems.

# References