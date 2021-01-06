import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
from xgboost.sklearn import XGBRegressor
import pickle
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_log_error
from sklearn.preprocessing import MinMaxScaler

# Get training data
df_train = pd.read_csv('.../data-analysis/preprocessed_train_data.csv', engine='python')
target = 'Num of Profile Likes'
# Observations and labels matrix
X = MinMaxScaler().fit_transform(df_train.drop([target], axis=1))
y = df_train[target]

# Split train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=10, shuffle=True)

# Set up param grid
param_grid = {'nthread': [4],  # when use hyperthread, xgboost may become slower
              'objective': ['reg:squarederror'],
              'learning_rate': [.03, 0.05, .07],  # so called `eta` value
              'max_depth': [5, 6, 7],
              'min_child_weight': [4],
              'silent': [1],
              'subsample': [0.7],
              'colsample_bytree': [0.7],
              'n_estimators': [200, 500, 700]}

# Set up grid search with cross validation
print("Preparing search grid!")
model = model = XGBRegressor()
search_grid = GridSearchCV(model, param_grid, scoring='neg_root_mean_squared_error', n_jobs=-1, cv=10, refit=True,
                           verbose=True)
# Fit model
print("Start fitting model!")
search_grid.fit(X_train, y_train)

# Return best estimators
print("The best parameters found are", search_grid.best_params_)
print("The best RMSLE score was", search_grid.best_score_)

predictor = search_grid.best_estimator_
y_predicted = np.exp(predictor.predict(X_test))
rmsle = np.sqrt(mean_squared_log_error(np.exp(y_test), y_predicted))
print("The RMSLE on the test set is ", rmsle)

xgb_best_results_txt = 'xgb_best_results.txt'
with open(xgb_best_results_txt, 'w') as f:
    print(search_grid.best_params_, file=f)
    print(search_grid.best_score_, file=f)

xgb_pickle = 'xgb_pickle.pkl'
with open(xgb_pickle, 'wb') as file:
    pickle.dump(predictor, file)
