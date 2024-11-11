import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
from datetime import datetime

from App import StreamlitUI

def setup_page_config():
    """Configure the Streamlit page"""
    st.set_page_config(
        page_title="Model Implementation",
        page_icon="🤖",
        layout="wide"
    )

def load_model_and_scaler(model_file, scaler_file):
    try:
        # Create a temporary directory if it doesn't exist
        temp_dir = 'temp_uploads'
        if not os.path.exists(temp_dir):
            os.makedirs(temp_dir)
        
        # Generate unique filenames
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        temp_model_path = os.path.join(temp_dir, f'model_{timestamp}.pkl')
        temp_scaler_path = os.path.join(temp_dir, f'scaler_{timestamp}.pkl')
        
        # Save uploaded files
        with open(temp_model_path, 'wb') as f:
            f.write(model_file.getbuffer())
        with open(temp_scaler_path, 'wb') as f:
            f.write(scaler_file.getbuffer())
        
        # Load the files using pickle
        with open(temp_model_path, 'rb') as f:
            model = pickle.load(f)
        with open(temp_scaler_path, 'rb') as f:
            scaler = pickle.load(f)
        
        # Clean up
        os.remove(temp_model_path)
        os.remove(temp_scaler_path)
        
        return model, scaler
    except Exception as e:
        st.error(f"Error loading model or scaler: {str(e)}")
        return None, None

def predict(model, scaler, features):
    try:
        # Convert features to numpy array and reshape
        features_array = np.array(features).reshape(1, -1)
        
        # Scale features
        features_scaled = scaler.transform(features_array)
        
        # Make prediction
        prediction = model.predict(features_scaled)
        
        # Get prediction probabilities if available
        try:
            probabilities = model.predict_proba(features_scaled)
            return prediction[0], probabilities[0]
        except:
            return prediction[0], None
            
    except Exception as e:
        st.error(f"Error making prediction: {str(e)}")
        return None, None

def generate_random_features(feature_names):
    """Generate random but realistic values for features"""
    random_values = {}
    
    # Get ranges from default configs in App.py
    feature_ranges = {}
    for feature_name in feature_names:
        min_val = float('inf')
        max_val = float('-inf')
        
        # Calculate min/max across all classes in default configs
        for class_config in StreamlitUI().default_configs.values():
            mean = class_config['mean']
            std = class_config['std']
            
            # Get index of matching feature
            try:
                idx = StreamlitUI().default_features.index(feature_name)
                feature_min = mean[idx] - 3*std[idx]  # 3 std deviations for 99.7% coverage
                feature_max = mean[idx] + 3*std[idx]
                
                min_val = min(min_val, feature_min)
                max_val = max(max_val, feature_max)
            except ValueError:
                continue
                
        # If feature not found in defaults, use reasonable fallback range
        if min_val == float('inf'):
            min_val, max_val = 0, 100
            
        feature_ranges[feature_name] = (min_val, max_val)
    
    for feature in feature_names:
        # Default range if feature not in predefined ranges
        min_val, max_val = 0, 100
        
        # Check if any of the known features are in the feature name
        for key, (min_range, max_range) in feature_ranges.items():
            if key.lower() in feature.lower():
                min_val, max_val = min_range, max_range
                break
                
        random_values[feature] = round(np.random.uniform(min_val, max_val), 2)
    
    return random_values

def show():
    st.title("Model Implementation")
    
    # Initialize session state for random values if not exists
    if 'random_values' not in st.session_state:
        st.session_state.random_values = {}
    
    # Keep file uploaders in sidebar
    st.sidebar.subheader("Upload Model Files")
    model_file = st.sidebar.file_uploader("Upload Model (.pkl)", type=['pkl'])
    scaler_file = st.sidebar.file_uploader("Upload Scaler (.pkl)", type=['pkl'])

    # Only proceed if both files are uploaded
    if model_file and scaler_file:
        model, scaler = load_model_and_scaler(model_file, scaler_file)

        if model and scaler:
            st.sidebar.success("Model and scaler loaded successfully!")

            # Get feature names from scaler
            feature_names = None
            if hasattr(scaler, 'feature_names_in_'):
                feature_names = scaler.feature_names_in_
            elif hasattr(model, 'feature_names_in_'):
                feature_names = model.feature_names_in_

            if feature_names is None:
                feature_names_input = st.sidebar.text_input(
                    "Enter feature names (comma-separated)",
                    "feature1, feature2, feature3"
                )
                feature_names = [f.strip() for f in feature_names_input.split(",")]
                st.sidebar.info("Feature names were not found in the model/scaler. Using manually entered names.")

            # Create two main columns for the page layout
            input_col, result_col = st.columns(2)

            # Left column for feature inputs
            with input_col:
                st.subheader("Enter Feature Values")
                
                # Add randomization button
                col1, col2 = st.columns([1, 2])
                with col1:
                    if st.button("🎲 Randomize"):
                        # Generate new random values
                        st.session_state.random_values = generate_random_features(feature_names)
                        # Update session state for each feature
                        for feature in feature_names:
                            st.session_state[f"input_{feature}"] = st.session_state.random_values[feature]
                with col2:
                    st.markdown("<div style='margin-top: 8px;'>Generate realistic random values</div>", 
                              unsafe_allow_html=True)
                
                # Create feature inputs in a grid layout
                feature_values = {}
                input_cols = st.columns(2)  # 2 columns for feature inputs
                for idx, feature in enumerate(feature_names):
                    with input_cols[idx % 2]:
                        # Initialize session state for this input if not exists
                        if f"input_{feature}" not in st.session_state:
                            st.session_state[f"input_{feature}"] = 0.0
                        
                        feature_values[feature] = st.number_input(
                            f"{feature}",
                            key=f"input_{feature}",
                            step=1.0,
                            format="%.2f"
                        )

                # Make prediction button
                predict_clicked = st.button("Make Prediction")

            # Right column for prediction results
            with result_col:
                st.subheader("Prediction Results")
                
                # Make prediction when values are available or button is clicked
                if predict_clicked or st.session_state.random_values:
                    # Prepare features in correct order
                    features = [feature_values[feature] for feature in feature_names]
                    
                    # Get prediction
                    prediction, probabilities = predict(model, scaler, features)
                    
                    if prediction is not None:
                        st.write(f"Predicted Class: **{prediction}**")
                        
                        # Display probabilities if available
                        if probabilities is not None:
                            st.write("Class Probabilities:")
                            prob_df = pd.DataFrame({
                                'Class': model.classes_,
                                'Probability': probabilities
                            })
                            
                            # Display as bar chart
                            st.bar_chart(
                                prob_df.set_index('Class')
                            )
                else:
                    st.info("Enter feature values and click 'Make Prediction' to see results.")
    else:
        st.sidebar.info("Please upload both model and scaler files to proceed.")


if __name__ == "__main__":
    setup_page_config()
    show()