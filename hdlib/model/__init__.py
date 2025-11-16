# This makes the classes directly available when importing the 'model' package
# Supervised classification model (classic and quantum implementations) and feature selection
from .classification import ClassificationModel, QuantumClassificationModel

# Graph encoding and edges prediction
from .graph import GraphModel

# Regression model
from .regression import RegressionEncoder, RegressionModel

# Unsupervised learning model / clustering
from .clustering import ClusteringModel