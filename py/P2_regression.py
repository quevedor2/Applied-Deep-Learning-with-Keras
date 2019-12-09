

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



#################################
#### Case Study - Regression ####
# We will be using the house sales dataset from King County, WA on Kaggle: https://www.kaggle.com/harlfoxem/housesalesprediction
# The data has around 21,000 rows with 20 features. The value we're tring to predict is a floating point number labeld as "price".

############################################
#### Data Visualization & Preprocessing ####
datadir=Path('~/git/Applied-Deep-Learning-with-Keras')
rawdf = pd.read_csv(datadir / 'data' / 'kc_house_data.csv')
rawdf.head()


## Describe dataset
rawdf.describe()
rawdf.hist(figsize=(10, 10))
plt.tight_layout()

plt.figure(figsize=(6, 8))
sns.heatmap(rawdf.corr()[['price']], annot=True, vmin=-1, vmax=1)

## Scale features
df = rawdf.copy()

ss = StandardScaler()
scale_features = ['bathrooms', 'bedrooms', 'grade', 'sqft_above',
                  'sqft_basement', 'sqft_living', 'sqft_living15', 'sqft_lot', 'sqft_lot15']
df[scale_features] = ss.fit_transform(df[scale_features])

## Bucketization of features
bucketized_features = ['yr_built', 'yr_renovated', 'lat', 'long']

bins = range(1890, 2021, 10)
df['yr_built'] = pd.cut(df.yr_built, bins, labels=bins[:-1])

bins = np.append([-10], np.arange(1930, 2021, 10))
df['yr_renovated'] = pd.cut(df.yr_renovated, bins, labels=bins[:-1])

bins = np.arange(47.00, 47.90, 0.05)
df['lat'] = pd.cut(df.lat, bins, labels=bins[:-1])

bins = np.arange(-122.60, -121.10, 0.05)
df['long'] = pd.cut(df.long, bins, labels=bins[:-1])

## One-hot encoding of categorical features
df['date'] = [datetime.strptime(x, '%Y%m%dT000000').strftime('%Y-%m') for x in rawdf['date'].values]
df['zipcode'] = df['zipcode'].astype(str)
categorical_features = ['zipcode', 'date']
categorical_features.extend(bucketized_features)
df_cat = pd.get_dummies(df[categorical_features])
df = df.drop(categorical_features, axis=1)
df = pd.concat([df, df_cat], axis=1)

## drop features
drop_features = ['id']
df = df.drop(drop_features, axis=1)

df.head()


## Visualization: Feature correlations with price
plt.figure(figsize=(4, 8))
tempdf = df.corr()[['price']].sort_values('price', ascending=False).iloc[:20, :]
sns.heatmap(tempdf, annot=True, vmin=-1, vmax=1)


########################
#### Model Building ####
X = df.drop(['price'], axis=1).values
y = df['price'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)

## Winsorization of y-values
factor = 5
y_train[np.abs(y_train - y_train.mean()) > (factor * y_train.std())] = y_train.mean() + factor*y_train.std()

## scale the price
ss_price = StandardScaler()
y_train = ss_price.fit_transform(y_train.reshape(-1, 1))
y_test = ss_price.transform(y_test.reshape(-1,1))

####################
#### Dumb Model ####
train_prices = ss_price.inverse_transform(y_train)
dumb_prices = np.zeros(real_prices.shape)
dumb_prices.fill(train_prices.mean())
dumb_error = mean_absolute_error(real_prices, dumb_prices)
print('Dumb model error:', output_dollars(dumb_error))


#################################
#### Linear Regression Model ####
M = RegressionModels(X)
M.linear_model()

linr_history = M.model.fit(X_train, y_train, epochs=30, verbose=0, validation_split=0.2)
plot_loss(linr_history)

## Performance metrics
M.model.evaluate(X_test, y_test, verbose=0)

## Get coefficients of linear model
linr_wdf = pd.DataFrame(M.model.get_weights()[0].T,
                      columns=df.drop(['price'], axis=1).columns).T.sort_values(0, ascending=False)
linr_wdf.columns = ['feature_weight']
linr_wdf.iloc[:20,:]

## Get predictions and error
linr_predictions = M.model.predict(X_test).ravel()
linr_prices = ss_price.inverse_transform(linr_predictions)
linr_error = mean_absolute_error(real_prices, linr_prices)
print('Linear model error:', output_dollars(linr_error))

###############################
#### Deep Regression Model ####
# The loss is behaving incosistently between runs, this one was good but the previous ones were bad. Needs debugging.
M = RegressionModels(X)
M.ann_model()


deep_history = M.model.fit(X_train, y_train, epochs=30, verbose=0, validation_split=0.2)
plot_loss(deep_history)
M.model.evaluate(X_test, y_test, verbose=0)

## with early stopping callback
early_stop = EarlyStopping(monitor='val_loss', patience=2, verbose=1)
deep_early = M.model.fit(X_train, y_train, epochs=30, verbose=0, validation_split=0.2,
                        callbacks=[early_stop])
plot_loss(deep_early)
M.model.evaluate(X_test, y_test, verbose=0)


## Performance metrics
plot_compare_histories([linr_history, deep_history, deep_early],
                        ['Linear Reg', 'Deep ANN', 'Deep ANN (early)'],
                        plot_accuracy=False)

def output_dollars(num):
    return '$'+str("{:,}".format(int(num)))

print('Average house price:', output_dollars(rawdf['price'].mean()))

real_prices = ss_price.inverse_transform(y_test)

## Get predictions and error
deep_predictions = M.model.predict(X_test).ravel()
deep_prices = ss_price.inverse_transform(deep_predictions)
deep_error = mean_absolute_error(real_prices, deep_prices)
print('Deep model error:', output_dollars(deep_error))

###################################
#### Assemble Prediction Error ####
tdf = pd.DataFrame([['Naive Model', output_dollars(dumb_error)],
                    ['Linear Regression', output_dollars(linr_error)],
                    ['Deep ANN', output_dollars(deep_error)]],
                   columns=['Model', 'Price Error'])
tdf

print(r2_score(real_prices, dumb_prices), r2_score(real_prices, linr_prices), r2_score(real_prices, deep_prices))
