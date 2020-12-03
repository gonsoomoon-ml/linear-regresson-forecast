import joblib, os

def predict_fn(input_data, model):
    """Preprocess input data
    """
    payload = input_data
    
    predictions = model.predict(payload)
    print("From user-inference file- Shape of predictions: ", predictions.shape)

    return predictions

def model_fn(model_dir):
    """
    Deserialized and return fitted model
    Note that this should have the same name as the serialized model in the main method
    """   
    model = joblib.load(os.path.join(model_dir, "model.joblib"))
    print("From user-inference file- Model loaded: ")     
    
    return model



