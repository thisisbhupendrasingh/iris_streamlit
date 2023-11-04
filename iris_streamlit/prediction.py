import joblib

def predict(data):
    model = joblib.load("iris_streamlit/rf_model.sav")
    answer = model.predict(data)
    return answer