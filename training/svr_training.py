from sklearn.svm import SVR
from training import trainer

# Set up trainer which initializes train/test sets
trainer = trainer.Trainer()
model = SVR()

# Set up grid for hyperparameter tuning
param_grid = [
    {'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
     'degree': [2, 3, 4],
     'gamma': ['scale', 'auto'],
     'C': [0.1, 10, 20]
     }
]
# Train
trained_grid = trainer.train(model, param_grid)
predictor = trained_grid.best_estimator_
best_params = trained_grid.best_params_
best_score = trained_grid.best_score_
print("The best parameters found are", best_params)
print("The best RMSE score was", best_score)

# Test
trainer.test(predictor)

# Save results
params_file = 'ridge.txt'
pickle_file = 'ridge.pkl'
trainer.save(predictor, best_params, best_score, params_file, pickle_file)
