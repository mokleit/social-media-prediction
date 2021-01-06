import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import pickle
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import mean_squared_error, mean_squared_log_error
from sklearn.preprocessing import MinMaxScaler

# Get training data
df_train = pd.read_csv('../data-analysis/preprocessed_train_data.csv', engine='python')
target = 'Num of Profile Likes'
# Observations and labels matrix
X = MinMaxScaler().fit_transform(df_train.drop([target], axis=1))
y = df_train[target]

# Split train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=10, shuffle=True)

# Set up param grid
param_grid = {
    'learning_rate': [0.01, 0.1, 0.5, 1],
    'n_estimators': [200, 350, 500, 750, 1000],
    'num_leaves': [20, 30, 40, 60],
    'max_depth': [2, 5, 10],
    'min_child_weight': [0.01, 1, 2],
    'colsample_bytree': [0.01, 0.1, 0.5, 1],
}

# Set up grid search with cross validation
print("Preparing search grid!")
model = lgb.LGBMRegressor(subsample=0.9)
search_grid = GridSearchCV(model, param_grid, scoring='neg_mean_squared_log_error', n_jobs=-1, cv=10, refit=True)

# Fit model
print("Start fitting model!")
search_grid.fit(X_train, y_train)

# Return best estimators
print("The best parameters found are", search_grid.best_params_)
print("The best RMSLE score was", search_grid.best_score_)

predictor = search_grid.best_estimator_
y_predicted = predictor.predict(X_test)
rmsle = np.sqrt(mean_squared_error(y_test, y_predicted))
print("The RMSLE on the test set is ", rmsle)

lgbm_best_results_txt = 'lgbm_best_results.txt'
with open(lgbm_best_results_txt, 'w') as f:
    print(search_grid.best_params_, file=f)
    print(search_grid.best_score_, file=f)

lgbm_pickle = 'lgbm_model.pkl'
with open(lgbm_pickle, 'wb') as file:
    pickle.dump(predictor, file)
