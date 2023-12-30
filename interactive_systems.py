```python
# Importing necessary libraries
import streamlit as st
from ai_models import AI_Models
from utils import load_data, preprocess_data, split_data

class Interactive_Systems:
    """
    Class to handle interactive systems
    """
    def __init__(self, file_path):
        self.file_path = file_path
        self.data = load_data(file_path)
        self.processed_data = preprocess_data(self.data)
        self.X_train, self.X_test, self.y_train, self.y_test = split_data(self.processed_data)

    def interactive_model_selection(self):
        """
        Function to allow users to select and interact with different AI models
        """
        st.title('AI Maverick Interactive Systems')
        st.write('Welcome to AI Maverick! Here, you can interact with different AI models and see their performance.')
        
        model_names = ["Random Forest", "Logistic Regression", "Support Vector Machine", "Neural Network", 
                       "K-Nearest Neighbors", "Decision Tree", "Naive Bayes", "Gradient Boosting", 
                       "XGBoost", "LightGBM", "CatBoost"]
        
        selected_model = st.selectbox('Select a model to train and evaluate:', model_names)
        
        if st.button('Train and Evaluate'):
            ai_models = AI_Models(self.X_train, self.X_test, self.y_train, self.y_test)
            model = ai_models.models[selected_model]
            st.write(f"Training {selected_model}...")
            model.fit(self.X_train, self.y_train)
            st.write(f"{selected_model} trained successfully.")
            st.write(f"Evaluating {selected_model}...")
            evaluate_model(model, self.X_test, self.y_test)
            st.write(f"{selected_model} evaluated successfully.")
            
if __name__ == "__main__":
    interactive_systems = Interactive_Systems('your_data_file_path.csv')
    interactive_systems.interactive_model_selection()
```

