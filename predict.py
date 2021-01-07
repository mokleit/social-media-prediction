import numpy as np
import pandas as pd
import pickle
from sklearn.preprocessing import MinMaxScaler

MODEL_PATH = 'training/estimators/'
PREDICTIONS_PATH = 'predictions/'
MODEL = 'lgbm'

models = {
    'lasso': MODEL_PATH+'lasso.pkl',
    'lgbm': MODEL_PATH+'lgbm.pkl',
    'xgb': MODEL_PATH+'xgb.pkl'
}

# Load data
data = pd.read_csv('data/preprocessed_test_data.csv')
Id = data['Id']
X = MinMaxScaler().fit_transform(data.drop(['Id'], axis=1))

# Load model
model = pickle.load(open(models[MODEL], 'rb'))

# Predict
predictions = np.exp(model.predict(X))
submission = pd.DataFrame({'Id': Id, 'Predicted': predictions})
submission.to_csv(PREDICTIONS_PATH + MODEL + "_predictions.csv", index=False)
print("Submissions saved at", PREDICTIONS_PATH + MODEL + "_predictions.csv")
