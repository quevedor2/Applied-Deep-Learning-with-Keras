
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



############################################
#### Case Study - Binary Classification ####

# We will be using the Human Resources Analytics dataset on Kaggle: https://www.kaggle.com/ludobenistant/hr-analytics
# We're trying to predict whether an employee will leave based on various features such as number of projects they worked on, time spent at the company, last performance review, salary etc. The dataset has around 15,000 rows and 9 columns. The column we're trying to predict is called "left". It's a binary column with 0/1 values. The label 1 means that the employee has left.

############################################
#### Data Visualization & Preprocessing ####

datadir=Path('~/git/Applied-Deep-Learning-with-Keras')
rawdf = pd.read_csv(datadir / 'data' / 'HR.csv')

# Descriptors
rawdf.head()
rawdf.info()
rawdf.describe()
(rawdf['left'].value_counts())/rawdf['left'].count()


## Correlation Plots
# Correlation of features with "Left" (left the company)
plt.figure(figsize=(5, 5))
sns.heatmap(rawdf.corr()[['left']], annot=True, vmin=-1, vmax=1)

# All-by-all correlation of features
plt.figure(figsize=(10, 8))
sns.heatmap(rawdf.corr(), annot=True, square=True, vmin=-1, vmax=1)


## Histogram of feature data
rawdf.hist(figsize=(10, 8))
plt.tight_layout()


## Scale the features to 0-1 scale
df = rawdf.copy()

ss = StandardScaler()
scale_features = ['average_monthly_hours', 'number_project', 'time_spend_company']
df[scale_features] = ss.fit_transform(df[scale_features])

## One-hot encoding categorial features
categorical_features = ['sales', 'salary']
df_cat = pd.get_dummies(df[categorical_features])
df = df.drop(categorical_features, axis=1)
df = pd.concat([df, df_cat], axis=1)
df.head()

## Separate dataframe into our X and Y; store as numpy array

X = df.drop('left', axis=1).values # (14999, 20)
y = df['left'].values # (14999,)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
print(X_train.shape, y_train.shape, X_test.shape, y_test.shape) # (10499, 20) (10499,) (4500, 20) (4500,)


###################################
#### Logistic Regression Model ####
M=BinaryModels(X_train)
M.log_model()

lr_history = M.model.fit(X_train, y_train, verbose=0, epochs=30)
plot_loss_accuracy(lr_history)


## Performance metrics
y_pred = M.model.predict_classes(X_test, verbose=0)
print(classification_report(y_test, y_pred))
plot_confusion_matrix(M.model, X_test, y_test)


## Keras cross validation
new_X, new_y = shuffle(X, y, random_state=0)
model = KerasClassifier(build_fn=get_model, epochs=5, verbose=0)
scores = cross_val_score(model, new_X, new_y, cv=5)
print("Accuracy: %0.2f%% (+/- %0.2f%%)" % (100*scores.mean(), 100*scores.std()*2))


##########################
#### Deep Model (ANN) ####
M = BinaryModels(X_train)
M.ann_model()

deep_history = M.model.fit(X_train, y_train, verbose=0, epochs=30)
plot_loss_accuracy(deep_history)


## Performance metrics
plot_compare_histories([lr_history, deep_history], ['Logistic Reg', 'Deep ANN'])
y_pred = M.model.predict_classes(X_test, verbose=0)
print(classification_report(y_test, y_pred))
plot_confusion_matrix(deep_model, X_test, y_test)

## Keras cross validation
model = KerasClassifier(build_fn=get_model, epochs=5, verbose=0)
scores = cross_val_score(model, X, y, cv=5)
print("Accuracy: %0.2f%% (+/- %0.2f%%)" % (100*scores.mean(), 100*scores.std()*2))


##################################
#### Deep Model Visualization ####

M = BinaryModels(X_train)
M.ann_vis_model()

history = M.model.fit(X_train, y_train, verbose=0, epochs=10)
plot_loss_accuracy(history)


## input to 2-D mapping
inp1 = M.model.layers[0].input
out1 = M.model.layers[2].output
func1 = K.function([inp1], [out1])

## 2-D to score prediction
inp2 = M.model.layers[3].input
out2 = M.model.layers[3].output
func2 = K.function([inp2], [out2])

features = func1([X_test])[0]
plot_decision_boundary(lambda x: func2([x])[0], features, y_test)
plt.title('Test Data Separation')

y_pred = M.model.predict_classes(X_test, verbose=0)
print(classification_report(y_test, y_pred))
plot_confusion_matrix(M.model, X_test, y_test)
