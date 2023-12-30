```python
# Importing necessary libraries
import time
import numpy as np
from utils import load_data, preprocess_data, split_data
from ai_models import AI_Models

class RealTimeAnalytics:
    """
    Class to handle real-time analytics
    """
    def __init__(self, file_path):
        self.file_path = file_path
        self.data = load_data(file_path)
        self.preprocessed_data = preprocess_data(self.data)
        self.X_train, self.X_test, self.y_train, self.y_test = split_data(self.preprocessed_data)

    def real_time_prediction(self, model_name):
        """
        Function to make real-time predictions with a specific model
        """
        ai_models = AI_Models(self.X_train, self.X_test, self.y_train, self.y_test)
        model = ai_models.models[model_name]
        model.fit(self.X_train, self.y_train)
        
        while True:
            try:
                # Load the latest data
                latest_data = load_data(self.file_path)
                latest_preprocessed_data = preprocess_data(latest_data)
                X_latest = latest_preprocessed_data.drop('target', axis=1)
                
                # Make a prediction on the latest data
                prediction = model.predict(X_latest)
                print(f"Real-time prediction from {model_name}: {prediction}")
                
                # Wait for a short period of time before making the next prediction
                time.sleep(5)
                
            except KeyboardInterrupt:
                print("Real-time prediction stopped.")
                break
```

