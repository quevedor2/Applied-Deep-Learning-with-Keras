

#################
#### MODULES ####

##get_ipython().magic('matplotlib inline')
##get_ipython().magic("config InlineBackend.figure_format = 'retina'")
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
import warnings
from pathlib import Path

warnings.filterwarnings('ignore')
pd.options.display.float_format = '{:,.2f}'.format
pd.set_option('display.max_rows', 100)
pd.set_option('display.max_columns', 200)
pd.set_option('display.width', 250)

from __future__ import print_function
from datetime import datetime
from matplotlib.colors import ListedColormap
from sklearn.datasets import make_classification, make_moons, make_circles
from sklearn.metrics import confusion_matrix, classification_report, mean_squared_error, mean_absolute_error, r2_score
from sklearn.linear_model import LogisticRegression
from sklearn.utils import shuffle
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Activation
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.python.keras.utils.np_utils import to_categorical
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold, KFold
import tensorflow.keras.backend as K
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier

import os
os.chdir("/cluster/home/quever/git/Applied-Deep-Learning-with-Keras/py")
from models import *
from P2_utility_functions import *



################################################
#### Case Study - MultiClass Classification ####
# We will be using the well known Iris dataset.


############################################
#### Data Visualization & Preprocessing ####
datadir=Path('~/git/Applied-Deep-Learning-with-Keras')
df = pd.read_csv(datadir / 'data' / 'iris.csv')
df.sample(n=5)



## Pair-wise distribution of features
sns.pairplot(df, hue='label')


## Feature scaling and organizing X and y
X = df.values[:, :-1]
ss = StandardScaler()
X = ss.fit_transform(X)
y = pd.get_dummies(df['label']).values
print(X.shape, y.shape)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
print(X_train.shape, y_train.shape)


## Building softmax regression models
M = MulticlassModels(X)
M.sr_model()

history = M.model.fit(X_train, y_train, epochs=30, verbose=0)
plot_loss_accuracy(history)

## Performance metrics
y_pred_class = M.model.predict_classes(X_test, verbose=0)
y_test_class = np.argmax(y_test, axis=1)
print(classification_report(y_test_class, y_pred_class))
plot_confusion_matrix(M.model, X_test, y_test_class)


## Building deep ANN model
M = MulticlassModels(X)
M.ann_model()

history = M.model.fit(X_train, y_train, epochs=100, verbose=0)
plot_loss_accuracy(history)

## Performance metrics
y_pred_class = M.model.predict_classes(X_test, verbose=0)
y_test_class = np.argmax(y_test, axis=1)
print(classification_report(y_test_class, y_pred_class))
plot_confusion_matrix(M.model, X_test, y_test_class)


## 5-fold Cross-Validation
cv = StratifiedKFold(n_splits=5, random_state=0)

M = MulticlassModels(X)
M.sr_model()
lin_model = KerasClassifier(build_fn=get_model, epochs=50, verbose=0)
lin_scores = cross_val_score(lin_model, X, df['label'].values, cv=cv)
print("Accuracy: %0.2f%% (+/- %0.2f%%)" % (100*lin_scores.mean(), 100*lin_scores.std()*2))

M.ann_model()
deep_model = KerasClassifier(build_fn=get_model, epochs=50, verbose=0)
deep_scores = cross_val_score(deep_model, X, df['label'].values, cv=cv)
print("Accuracy: %0.2f%% (+/- %0.2f%%)" % (100*deep_scores.mean(), 100*deep_scores.std()*2))


## Visualize cross-validation
#f = plt.figure()
sns.distplot(lin_scores, hist=False, label='linear model')
sns.distplot(deep_scores, hist=False, label='deep model')
#f.savefig("foo.pdf", bbox_inches='tight')
