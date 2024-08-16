import streamlit as st
import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Load the model and preprocessing objects
model = joblib.load('event_prediction_model.pkl')
label_encoder_location = joblib.load('label_encoder_location.pkl')
label_encoder_weather = joblib.load('label_encoder_weather.pkl')
scaler = joblib.load('scaler.pkl')
feature_columns = joblib.load('feature_columns.pkl')

# Function to load and preprocess data
def load_and_preprocess_data(uploaded_file):
    try:
        data = pd.read_csv(uploaded_file)
        data.columns = data.columns.str.strip()
        data.fillna(method='ffill', inplace=True)

        original_data = data.copy()  # Keep a copy of the original data to display
        data['Location'] = label_encoder_location.transform(data['Location'])
        data['Weather Conditions'] = label_encoder_weather.transform(data['Weather Conditions'])

        data = data[feature_columns]

        data = pd.DataFrame(scaler.transform(data), columns=feature_columns)

        return data, original_data
    except Exception as e:
        st.error(f"Error: {e}")
        return None, None

# Main function for Streamlit app
def main():
    # Custom theme settings
    indigo_bg = "#4a00e0"
    lighter_bg = "#7c3aed"
    text_color = "#f0f0f0"

    # Apply the custom theme
    st.set_page_config(page_title="Event Prediction App 🎉", layout="wide", page_icon=":chart_with_upwards_trend:")
    st.markdown(f"""
        <style>
            .reportview-container .main .block-container{{
                max-width: 90%;
                padding-top: 2rem;
                padding-right: 2rem;
                padding-left: 2rem;
                padding-bottom: 2rem;
                background-color: {indigo_bg};
                border-radius: 10px;
            }}
            .sidebar .sidebar-content {{
                background-color: {lighter_bg};
                color: {text_color};
                border-radius: 10px;
            }}
            .block-container {{
                color: {text_color};
            }}
            .sidebar .sidebar-content .block-container {{
                color: {text_color};
            }}
            .btn-primary {{
                background-color: {lighter_bg};
                color: {text_color};
                font-weight: bold;
                border-radius: 0.5rem;
                padding: 10px 20px;
            }}
            .btn-primary:hover {{
                background-color: #6a00e0;
            }}
            .title {{
                font-size: 2.5rem;
                color: #ffffff;
                text-align: center;
                padding-bottom: 20px;
            }}
            .subtitle {{
                font-size: 1.5rem;
                color: #ffffff;
                text-align: center;
                padding-bottom: 20px;
            }}
        </style>
    """, unsafe_allow_html=True)

    st.markdown("<div class='title'>🎉 Event Prediction App 🎉</div>", unsafe_allow_html=True)
    
    # Sidebar with file upload and manual prediction options
    st.sidebar.markdown("<div class='subtitle'>🔍 Options 🔍</div>", unsafe_allow_html=True)
    uploaded_file = st.sidebar.file_uploader("Upload a CSV file", type="csv", help="Upload your event data CSV file here.")
    predict_manually = st.sidebar.checkbox("Predict Manually")

    # Load sample data to get unique values for dropdowns
    try:
        sample_data = pd.read_csv('ai_event_prediction.csv')
        sample_data.columns = sample_data.columns.str.strip()
        regions = sample_data['Location'].unique()
        weather_conditions = sample_data['Weather Conditions'].unique()
    except FileNotFoundError:
        st.error("Sample data file not found.")

    # Main content area
    if uploaded_file is not None:
        data, original_data = load_and_preprocess_data(uploaded_file)

        if original_data is not None:
            st.subheader("📄 Uploaded Data (Before Preprocessing)")
            st.write(original_data.head())

        if data is not None:
            st.subheader("📊 Uploaded Data (After Preprocessing)")
            st.write(data.head())

            if st.button('📈 Predict from Uploaded File'):
                predictions = model.predict(data)
                original_data['Predicted Event-Specific Factors'] = predictions
                st.subheader("🔮 Predictions")
                st.write(original_data)

                # Convert the DataFrame to CSV for download
                csv = original_data.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="💾 Download Predictions as CSV",
                    data=csv,
                    file_name='predictions.csv',
                    mime='text/csv'
                )

    if predict_manually:
        st.subheader("🔧 Predict Event-Specific Factors Manually")

        # Manual input form
        if 'sample_data' in locals():
            user_input = {}

            user_input['Location'] = st.selectbox("📍 Location", regions)
            user_input['Weather Conditions'] = st.selectbox("☁️ Weather Conditions", weather_conditions)
            other_columns = sample_data.columns.drop(['Event-Specific Factors', 'Date', 'Location', 'Weather Conditions'])

            for col in other_columns:
                user_input[col] = st.slider(f"{col}", min_value=0.0, max_value=1000.0, value=0.0, step=0.1)

            user_input_df = pd.DataFrame(user_input, index=[0])

            if st.button('🔮 Predict Manually'):
                try:
                    # Preprocess the manual input data
                    user_input_df['Location'] = label_encoder_location.transform(user_input_df['Location'])
                    user_input_df['Weather Conditions'] = label_encoder_weather.transform(user_input_df['Weather Conditions'])
                    user_input_df = user_input_df[feature_columns]

                    user_input_df = pd.DataFrame(scaler.transform(user_input_df), columns=feature_columns)

                    prediction = model.predict(user_input_df)[0]

                    # Determine the population range based on the prediction
                    if prediction == 'small population attended':
                        population_range = '0-499'
                    elif prediction == 'moderate population attended':
                        population_range = '500-999'
                    else:
                        population_range = '1000+'

                    st.markdown(f"<p style='color:black; font-weight:bold;'>Predicted Population Attended: {prediction}</p>", unsafe_allow_html=True)
                    st.markdown(f"<p style='color:black; font-weight:bold;'>Population Range: {population_range}</p>", unsafe_allow_html=True)
                except Exception as e:
                    st.error(f"Error predicting: {e}")

if __name__ == "__main__":
    main()
