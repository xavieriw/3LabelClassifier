## Example code
- quickstart_inference.py: find AI / Not AI / Other confidence for an example dataset located in your baseline data, formatted as a .csv
  - To use: `cd 3LabelClassifier/bin` `python quickstart_inference.py [baseline_data.csv] [trained_model.sav] [output_filename.csv]`
- quickstart_train.py: Trains your model. The best model is saved to `finalized_model_<ModelName>.sav` alongside performance `finalized_performance.sav` (ROC-AUC) of each model
  - To use: `cd 3LabelClassifier/bin` `python quickstart_train.py`
