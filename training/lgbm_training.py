import lightgbm as lgb
from training import trainer

# Set up trainer which initializes train/test sets
trainer = trainer.Trainer()
model = lgb.LGBMRegressor(subsample=0.9)

# Set up param grid
param_grid = {
    'learning_rate': [0.01, 0.1, 0.5, 1],
    'n_estimators': [200, 350, 500, 750, 1000],
    'num_leaves': [20, 30, 40, 60],
    'max_depth': [2, 5, 10],
    'min_child_weight': [0.01, 1, 2],
    'colsample_bytree': [0.01, 0.1, 0.5, 1],
}

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
params_file = 'lgbm.txt'
pickle_file = 'lgbm.pkl'
trainer.save(predictor, best_params, best_score, pickle_file, params_file)