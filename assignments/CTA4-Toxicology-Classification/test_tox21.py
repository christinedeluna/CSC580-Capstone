import numpy as np
np.random.seed(456)

import tensorflow as tf
tf.random.set_seed(456)

from rdkit import RDLogger
RDLogger.DisableLog("rdApp.*")

import matplotlib.pyplot as plt
import deepchem as dc

from sklearn.metrics import accuracy_score

_, (train, valid, test), _ = dc.molnet.load_tox21()
