# Examples

Here we maintain examples of Python scripts which show how to use _hdlib_ for building vector-symbolic architectures.

## chopin2

The [chopin2.py](https://github.com/cumbof/hdlib/blob/main/examples/chopin2.py) script is a reimplementation of [chopin2](https://github.com/cumbof/chopin2) which makes use of _hdlib_ for building a supervised classification model following the hyperdimensional computing paradigm.

_chopin2_ has been originally proposed for dealing with massive amount of samples and features and has been applied on DNA-Methylation data of cancer.

> Cumbo F, Cappelli E, Weitschek E. _A brain-inspired hyperdimensional computing approach for classifying massive dna methylation data of cancer_. Algorithms. 2020 Sep 17;13(9):233. [https://doi.org/10.3390/a13090233](https://doi.org/10.3390/a13090233)

As the original software, the _chopin2.py_ script only accepts numerical datasets in input. The first row must contain the header with the name of the features, while the first column contains the name of the samples and the last column the class labels.

> **Note:** Given an input matrix _M_, values in position _M(ij)_ must be numbers, except for the first row, first column, and last column.

### Available options

Here is a list of available options:

| Option                | Default | Mandatory | Description  |
|:----------------------|:--------|:---------:|:-------------|
| `--input`             |         | ⚑         | Path to the input matrix |
| `--fieldsep`          | `\t`    |           | Input field separator |
| `--dimensionality`    | `10000` |           | Vectors dimensionality |
| `--levels`            |         | ⚑         | Number of level vectors |
| `--kfolds`            | `0`     |           | Number of folds for cross-validating the classification model |
| `--test-percentage`   | `0`     |           | Percentage of data points for defining the test set |
| `--feature-selection` |         |           | Run the feature selection (`backward` or `forward` variable elimination) and report a ranking of features based on their importance |
| `--retrain`           | `0`     |           | Number of retraining iterations |
| `--threshold`         | `0.6`   |           | Threshold on the accuracy score. Used in conjunction with `--feature-selection` |
| `--uncertainty`       | `5.0`   |           | Uncertainty percentage. Used in conjunction with `--feature-selection` |
| `--nproc`             | `1`     |           | Make it parallel when possible |
| `--version`           |         |           | Print the tool version and exit |

> **Note:** The [chopin2\_iris.sh](https://github.com/cumbof/hdlib/blob/main/examples/chopin2_iris.sh) automatically downloads the [iris.csv](https://raw.githubusercontent.com/scikit-learn/scikit-learn/main/sklearn/datasets/data/iris.csv) dataset from the scikit-learn repository on GitHub and run _chopin2.py_ by using 10 level vectors and selecting features through the backward variable elimination technique in 5-folds cross-validation. Also note that this must be intended just as an example about how to run _chopin2.py_ and results strongly depend on the input dataset.
