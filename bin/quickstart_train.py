from hazdet.train import hyperparameter_tune_all_models
# If you have raw qualtrics data, we have helper functions to clean it
# e.g., X,y =  create_features(file='Tweet annotation.csv')

# This is your training data
filename ='../ground_truth_data/training_data_300.csv'
# saves best model as 'finalized_model_'+best_model+'.sav', 
# where best model could be NN (feed-forward neural network), RF (random forest), XGB (XGBoost), or SVM (support vector machine)
# Also saves performance (ROC-AUC) of each candidate model in a pickle file
hyperparameter_tune_all_models(filename)
