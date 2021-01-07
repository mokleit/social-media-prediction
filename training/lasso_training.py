from sklearn.linear_model import Lasso
from training import trainer

# Set up trainer which initializes train/test sets
trainer = trainer.Trainer()
model = Lasso(random_state=10)

# Set up param grid
param_grid = [
    {'alpha': [0, 0.001, 0.01, 0.05, 0.1, 1, 10],
     'fit_intercept': [False, True],
     'max_iter': [100, 300, 750, 1000, 2000],
     'positive': [True, False],
     'tol': [1e-4],
     'selection': ['cyclic', 'random']
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
params_file = 'lasso.txt'
pickle_file = 'lasso.pkl'
trainer.save(predictor, best_params, best_score, params_file, pickle_file)