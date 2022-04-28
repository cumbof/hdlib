# hdlib
Hyperdimensional Computing Library for building Vector Symbolic Architectures in Python 3.

<img src="https://anaconda.org/conda-forge/hdlib/badges/version.svg"> <img src="https://anaconda.org/conda-forge/hdlib/badges/downloads.svg">

## Install

It is available through `pip` and `conda`.
Please, use one of the following commands to start playing with `hdlib`:

```
# Install with pip
pip install hdlib

# Install with conda
conda install -c conda-forge hdlib
```

## Usage

The `hdlib` library provides two main modules, `space` and `arithmetic`. The first one contains constructors of `Space` and `Vector` objects that can be used to build vectors and the space that hosts them. The second module, calle `arithmetic`, contains a bunch of functions to operate on vectors.

```python
from hdlib.space import Space, Vector
from hdlib.arithmetic import bind, bundle, permute
```

### What is the dollar of Mexico?

This case example has been extracted from the following manuscript:
```
Kanerva, P., 2010, November. 
What we mean when we say" What's the dollar of Mexico?": Prototypes and mapping in concept space. 
In 2010 AAAI fall symposium series.
```

Let's say that we have the following information about two countries:

| Country (NAM)                       | Capital City (CAP)  | Monetary Unit (MON)  |
| :---------------------------------- |:--------------------| :--------------------|
| United States of America (USA)      | Washington DC (WDC) | US Dollar (DOL)      |
| Mexico (MEX)                        | Mexico City (MXC)   | Peso (PES)           |

Country information can be encoded in hyperdimensional vectors, one for each country:

```python
# Initialize vectors space
space = Space()

# Define features and country information
names = [
  "NAM", "CAP", "MON", # Features
  "USA", "WDC", "DOL", # United States of America
  "MEX", "MXC", "PES"  # Mexico
]

# Build a random bipolar vector for each feature and country information
# Add vectors to the space
space.bulkInsert(names)

# Encode USA information in a single vector
# USTATES = [(NAM * USA) + (CAP * WDC) + (MON * DOL)]
USTATES_NAM = bind(space.get("NAM"), space.get("USA")) # Bind NAM with USA
USTATES_CAP = bind(space.get("CAP"), space.get("WDC")) # Bind CAP with WDC
USTATES_MON = bind(space.get("MON"), space.get("DOL")) # Bind MON with DOL
USTATES = bundle(bundle(USTATES_NAM, USTATES_CAP), USTATES_MON) # Bundle USTATES_NAM, USTATES_CAP, and USTATES_MON

# Repeat the last step to encode MEX information in a single vector
# MEXICO = [(NAM * MEX) + (CAP * MXC) + (MON * PES)]
MEXICO_NAM = bind(space.get("NAM"), space.get("MEX")) # Bind NAM with MEX
MEXICO_CAP = bind(space.get("CAP"), space.get("MXC")) # Bind CAP with MXC
MEXICO_MON = bind(space.get("MON"), space.get("PES")) # Bind MON with PES
MEXICO = bundle(bundle(MEXICO_NAM, MEXICO_CAP), MEXICO_MON) # Bundle MEXICO_NAM, MEXICO_CAP, and MEXICO_MON
```

If we now pair `USTATES` with `MEXICO`, we get a bundle that pairs USA with Mexico, Washington DC with Mexico City, and US Dollar with Peso, plus noise.
```python
# F_UM = USTATES * MEXICO
#      = [(USA * MEX) + (WDC * MXC) + (DOL * PES) + noise]
F_UM = bind(USTATES, MEXICO)
```

In order to retrieve the Monetary Unit that in Mexico corresponds to Dollar:
```python
# DOL * F_UM = DOL * [(USA * MEX) + (WDC * MXC) + (DOL * PES) + noise]
#            = [(DOL * USA * MEX) + (DOL * WDC * MXC) + (DOL * DOL * PES) + (DOL * noise)]
#            = [noise1 + noise2 + PES + noise3]
#            = [PES + noise4]
#            ≈ PES
GUESS_PES = bind(space.get("DOL"), F_UM)
```

Finally answer the question by searching for the closest vector in space
```python
space.find(GUESS_PES)
```

## Contributing

Long-term discussion and bug reports are maintained via GitHub Issues, while code review is managed via GitHub Pull Requests.

Please, (i) be sure that there are no existing issues/PR concerning the same bug or improvement before opening a new issue/PR; (ii) write a clear and concise description of what the bug/PR is about; (iii) specifying the list of steps to reproduce the behavior in addition to versions and other technical details is highly recommended.
