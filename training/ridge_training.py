from sklearn.linear_model import Ridge
from training import trainer

# Set up trainer which initializes train/test sets
trainer = trainer.Trainer()
model = Ridge(random_state=10)

# Set up param grid
param_grid = [
    {'alpha': [0.1, 1, 10],
     'fit_intercept': [False, True],
     'max_iter': [100, 200, 500, 750, 1000, 1500],
     'solver': ['auto', 'svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag', 'saga']
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