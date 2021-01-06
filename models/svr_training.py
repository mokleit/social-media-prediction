import pickle
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_log_error
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import SelectKBest, f_regression

# Get training data
df_train = pd.read_csv('.../data-analysis/preprocessed_train_data.csv', engine='python')
target = 'Num of Profile Likes'
# Observations and labels matrix
X = MinMaxScaler().fit_transform(df_train.drop([target], axis=1))
# Use logarithm for tagret variable
# y = np.log1p(df_train[target])
y = df_train[target]

# Split train and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=10, shuffle=True)

# Set up grid for hyperparameter tuning
param_grid = [
    {'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
     'degree': [2, 3, 4],
     'gamma': ['scale', 'auto'],
     'C': [0.1, 10, 20]
     }
]

# Set up grid search with cross validation
print("Preparing search grid!")
search_grid = GridSearchCV(SVR(), param_grid, scoring='neg_root_mean_squared_error', n_jobs=-1, cv=10, refit=True)
# Fit model
print("Start fitting model!")
search_grid.fit(X_train, y_train)

# Return best estimators
print("The best parameters found are", search_grid.best_params_)
print("The best RMSLE score was", np.sqrt(search_grid.best_score_))

predictor = search_grid.best_estimator_
y_predicted = predictor.predict(X_test)
rmsle = np.sqrt(mean_squared_log_error(np.exp(y_test), y_predicted))
print("The RMSLE on the test set is ", rmsle)

svr_best_results_txt = 'svr_best_results.txt'
with open(svr_best_results_txt, 'w') as f:
    print(search_grid.best_params_, file=f)
    print(search_grid.best_score_, file=f)

# Save model to be reused by
svr_pkl = "svr_model.pkl"
with open(svr_pkl, 'wb') as file:
    pickle.dump(predictor, file)
