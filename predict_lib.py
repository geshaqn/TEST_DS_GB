from joblib import load as joblib_load

def load_pipeline(path='fitted_model.sav'):
    "Load fitted pipeline before predicting"
    return joblib_load(path)

def predict(pipeline, row):
    "Predict if motor is OK"
    if pipeline.predict([row[1:]])[0] < 0.8:
        return 'OK'
    return 'TO'
