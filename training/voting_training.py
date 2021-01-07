import pickle
import numpy as np
import pandas as pd
from sklearn.ensemble import VotingRegressor
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_log_error

# Get training data
df_train = pd.read_csv('../data/preprocessed_train_data.csv', engine='python')
target = 'Num of Profile Likes'
X = MinMaxScaler().fit_transform(df_train.drop([target], axis=1))
y = df_train[target]

# Split train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=10, shuffle=True)

lgbm = pickle.load(open('estimators/lgbm.pkl', 'rb'))
xgb = pickle.load(open('estimators/xgb.pkl', 'rb'))

model = VotingRegressor(
    estimators=[
        ('xgb', xgb),
        ('lgbm', lgbm),
    ])

# Calculate cross validation score for the Voting regresssor
scores = cross_val_score(model, X_train, y_train,
                         scoring="neg_root_mean_squared_error", cv=10, n_jobs=-1)

print("Voting: Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std()))

# Fit model
model.fit(X_train, y_train)
y_predicted = np.exp(model.predict(X_test))
rmsle = np.sqrt(mean_squared_log_error(np.exp(y_test), y_predicted))
print("The RMSLE on the test set is ", rmsle)

voting_pickle = 'estimators/voting.pkl'
with open(voting_pickle, 'wb') as file:
    pickle.dump(model, file)
