import numpy as np
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_log_error

default_data = {
    'data': '../data/preprocessed_train_data.csv',
    'target': 'Num of Profile Likes'
}


class Trainer:
    def __init__(self, data=default_data['data'], target=default_data['target'], test=0.2):
        # Get training data
        self.df_train = pd.read_csv(data, engine='python')
        self.target = target
        self.X = MinMaxScaler().fit_transform(self.df_train.drop([target], axis=1))
        self.y = self.df_train[target]
        # Split train and test sets
        self.X_train, self.X_test, self.y_train, self.y_test = \
            train_test_split(self.X, self.y, test_size=test, random_state=10, shuffle=True)

    def train(self, model, param_grid):
        search_grid = GridSearchCV(model, param_grid, scoring='neg_root_mean_squared_error', n_jobs=-1, cv=10,
                                   refit=True)
        # Fit model
        print("Start fitting model!")
        search_grid.fit(self.X_train, self.y_train)
        return search_grid

    def test(self, predictor):
        y_predicted = np.exp(predictor.predict(self.X_test))
        rmsle = np.sqrt(mean_squared_log_error(np.exp(self.y_test), y_predicted))
        print("The RMSLE on the test set is ", rmsle)

    @staticmethod
    def save(predictor, best_params, cv_score, params_name, pickle_name):
        params_path = 'parameters/' + params_name
        pickle_path = 'estimators/' + pickle_name
        with open(params_path, 'w') as f:
            print(best_params, file=f)
            print(cv_score, file=f)

        with open(pickle_path, 'wb') as file:
            pickle.dump(predictor, file)

