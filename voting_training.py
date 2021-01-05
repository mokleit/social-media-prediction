# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import pickle
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.ensemble import VotingRegressor
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_log_error

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
#Observations and labels matrix
X = MinMaxScaler().fit_transform(df_train.drop([target], axis=1))
#Use logarithm for tagret variable except for data exploration
y = df_train[target]

#Split train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=10, shuffle=True)

lgbm = pickle.load(open('../input/lgbm-training/lgbm_model.pkl', 'rb'))
xgb = pickle.load(open('../input/xgbregressor-training/xgb_pickle.pkl', 'rb'))

model = VotingRegressor(
    estimators=[
        ('xgb', xgb),
        ('lgbm', lgbm),
        ])


# Calculate cross validation score for the Voting regresssor
scores = cross_val_score(model, X_train, y_train,
                         scoring="neg_root_mean_squared_error", cv=10, n_jobs=-1)

print("Voting: Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std()))

#Fit model
model.fit(X_train, y_train)
y_predicted = np.exp(model.predict(X_test))
rmsle = np.sqrt(mean_squared_log_error(np.exp(y_test), y_predicted))
print("The RMSLE on the test set is ", rmsle)

voting_pickle = 'voting_pickle.pkl'
with open(voting_pickle, 'wb') as file:
    pickle.dump(model, file)