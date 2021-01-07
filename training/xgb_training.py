from xgboost.sklearn import XGBRegressor
from training import trainer

# Set up trainer which initializes train/test sets
trainer = trainer.Trainer()
model = XGBRegressor()

# Set up param grid
param_grid = {'nthread': [4],  # when use hyperthread, xgboost may become slower
              'objective': ['reg:squarederror'],
              'learning_rate': [.03, 0.05, .07],  # so called `eta` value
              'max_depth': [5, 6, 7],
              'min_child_weight': [4],
              'silent': [1],
              'subsample': [0.7],
              'colsample_bytree': [0.7],
              'n_estimators': [200, 500, 700]}

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
params_file = 'xgb.txt'
pickle_file = 'xgb.pkl'
trainer.save(predictor, best_params, best_score, pickle_file, params_file)