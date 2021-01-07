# SOCIAL MEDIA PREDICTION

We try to predict the number of likes associated with a certain social media profile. This is a **regression** problem. 
Everything was run on Kaggle.

## STRUCTURE
The datasets (raw and pre-processed) can be found [here](/data).
The data exploratory analysis as well as the pre-processing can be found in the following [notebook](/data-analysis/data-exploration.ipynb).
The pre-trained models can be found [here](/training/estimators).
The saved hyper-parameters for each model with the CV result can be found [here](/training/parameters).
The predictions as csv can be found [here](/predictions).

## HOW TO RUN

### USING PRE-TRAINED MODELS

1. Open the [predict](predict.py) script.
2. Based on the models dictionary in that script, change the **NAME** variable to reflect the model you want to use.
3. Execute the script.
4. The results are saved [here](/predictions)

### TRAIN THE MODEL FROM SCRATCH

1. Inside the [training](/training) folder, execute the model you want to train by choosing the file "modelname_training.py".
2. The trained model will be saved [here](/training/estimators)
3. Follow steps 1 through 4 of section "USING PRE_TRAINED MODELS".

