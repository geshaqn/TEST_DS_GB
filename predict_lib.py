from joblib import load as joblib_load

def load_pipeline(path='fitted_model.sav'):
    "Load fitted pipeline before predicting"
    return joblib_load(path)

def predict(pipeline, row):
    "Predict probability of motor breaking on current cycle"
    return pipeline.predict([row[1:]])
