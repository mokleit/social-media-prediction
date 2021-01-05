# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import pickle
from sklearn.preprocessing import MinMaxScaler

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


#Load test data
data = pd.read_csv('../input/data-exploration-ii/preprocessed_test_data.csv')
Id = data['Id']
X = MinMaxScaler().fit_transform(data.drop(['Id'], axis=1))

#Load model
filename = '../input/elasticnet-training/elasticnet_pickle.pkl'
model = pickle.load(open(filename, 'rb'))

#Predict
predictions = np.round(np.exp(model.predict(X)))
submission = pd.DataFrame({'Id': Id, 'Predicted': predictions})
submission.to_csv("submission.csv", index=False)
print("Finished submitting")