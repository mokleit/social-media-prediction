from sklearn.neighbors import KNeighborsRegressor
from training import trainer

# Set up trainer which initializes train/test sets
trainer = trainer.Trainer()
model = KNeighborsRegressor(n_jobs=-1)

# Set up param grid
param_grid = [
    {'n_neighbors': [2, 5, 7, 10, 15],
     'weights': ['uniform', 'distance'],
     'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
     'leaf_size': [10, 20, 30, 40],
     'p': [1, 2],
     'metric': ['euclidean', 'manhattan', 'chebyshev', 'minkowski', 'mahalanobis']
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
params_file = 'knn.txt'
pickle_file = 'knn.pkl'
trainer.save(predictor, best_params, best_score, params_file, pickle_file)