# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import pickle
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_log_error
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import Lasso
from sklearn.feature_selection import SelectKBest, chi2


# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

#Get training data
df_train = pd.read_csv('../input/data-exploration-ii/preprocessed_train_data.csv', engine='python')
target = 'Num of Profile Likes'
y = df_train[target]
#X = SelectKBest(chi2, k=10).fit_transform(df_train.drop([target], axis=1), y)
X = MinMaxScaler().fit_transform(df_train.drop([target], axis=1))
#Observations and labels matrix
#X = MinMaxScaler().fit_transform(df_train.drop([target], axis=1))
#Use logarithm for tagret variable except for data exploration
# y = np.log1p(df_train[target])


#Split train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=10, shuffle=True)

#Set up param grid
param_grid = [
    {'alpha': [0, 0.001, 0.01, 0.05, 0.1, 1, 10],
     'fit_intercept': [False, True],
     'max_iter': [100, 300, 750, 1000, 2000],
     'positive': [True, False],
     'tol': [1e-4],
     'selection': ['cyclic', 'random']
    }
]

#Set up grid search with cross validation
print("Preparing search grid!")
model = Lasso(random_state=10)
search_grid = GridSearchCV(model, param_grid, scoring='neg_root_mean_squared_error', n_jobs=-1, cv=10, refit=True)
#Fit model
print("Start fitting model!")
search_grid.fit(X_train, y_train)

#Return best estimators
print("The best parameters found are", search_grid.best_params_)
print("The best RMSLE score was", search_grid.best_score_)

predictor = search_grid.best_estimator_
y_predicted = np.exp(predictor.predict(X_test))
rmsle = np.sqrt(mean_squared_log_error(np.exp(y_test), y_predicted))
print("The RMSLE on the test set is ", rmsle)

lasso_best_results_txt = 'lasso_best_results.txt'
with open(lasso_best_results_txt, 'w') as f:
    print(search_grid.best_params_, file=f)
    print(search_grid.best_score_, file=f)

lasso_pickle = 'lasso_pickle.pkl'
with open(lasso_pickle, 'wb') as file:
    pickle.dump(predictor, file)