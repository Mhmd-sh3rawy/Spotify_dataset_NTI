
import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import pickle
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# Page config
st.set_page_config(
    page_title="🎵 Music Popularity Predictor",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
    }

    .prediction-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        box-shadow: 0 10px 30px rgba(0,0,0,0.2);
    }

    .high-popularity {
        background: linear-gradient(135deg, #4CAF50 0%, #45a049 100%);
    }

    .low-popularity {
        background: linear-gradient(135deg, #f44336 0%, #da190b 100%);
    }

    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #667eea;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)

# Cache model loading
@st.cache_resource
def load_model_and_scaler():
    """Load the trained model and scaler"""
    try:
        model = tf.keras.models.load_model('popularity_model.h5')

        with open('scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)

        with open('feature_names.pkl', 'rb') as f:
            feature_names = pickle.load(f)

        return model, scaler, feature_names, True
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None, None, False

# Load model
model, scaler, feature_names, model_loaded = load_model_and_scaler()

def main():
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🎵 Music Popularity Predictor</h1>
        <p>Predict whether your song will be a hit using advanced AI</p>
        <p>🚀 Running on Google Colab!</p>
    </div>
    """, unsafe_allow_html=True)

    if not model_loaded:
        st.error("⚠️ Model not loaded. Please ensure model files are present.")
        return

    # Sidebar
    st.sidebar.title("🎛️ Control Panel")
    st.sidebar.success("✅ Model Loaded Successfully")
    st.sidebar.info(f"📊 Features: {len(feature_names)}")

    # Input method selection
    input_method = st.sidebar.selectbox(
        "Choose Input Method",
        ["🎚️ Manual Input", "📁 Upload CSV", "🎲 Random Sample"]
    )

    if input_method == "🎚️ Manual Input":
        manual_input_interface()
    elif input_method == "📁 Upload CSV":
        csv_upload_interface()
    else:
        random_sample_interface()

def manual_input_interface():
    """Manual input interface with sliders"""
    st.subheader("🎚️ Adjust Song Features")

    # Create columns for better layout
    col1, col2 = st.columns(2)

    input_data = {}

    # Split features into two columns
    mid_point = len(feature_names) // 2

    with col1:
        st.markdown("### 🎵 Audio Features (Part 1)")
        for feature in feature_names[:mid_point]:
            input_data[feature] = st.slider(
                f"{feature.replace('_', ' ').title()}",
                min_value=-3.0,
                max_value=3.0,
                value=0.0,
                step=0.1,
                help=f"Adjust {feature} value"
            )

    with col2:
        st.markdown("### 🎼 Audio Features (Part 2)")
        for feature in feature_names[mid_point:]:
            input_data[feature] = st.slider(
                f"{feature.replace('_', ' ').title()}",
                min_value=-3.0,
                max_value=3.0,
                value=0.0,
                step=0.1,
                help=f"Adjust {feature} value"
            )

    # Prediction button
    if st.button("🔮 Predict Popularity", type="primary"):
        make_prediction(input_data)

    # Feature visualization
    if st.checkbox("📊 Show Feature Chart"):
        create_feature_chart(input_data)

def csv_upload_interface():
    """CSV upload interface"""
    st.subheader("📁 Upload CSV File")

    uploaded_file = st.file_uploader(
        "Choose a CSV file",
        type="csv",
        help="Upload a CSV file with song features"
    )

    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.success(f"✅ Loaded {len(df)} songs from CSV")

            # Show preview
            st.subheader("📋 Data Preview")
            st.dataframe(df.head())

            # Batch prediction
            if st.button("🔮 Predict All Songs", type="primary"):
                predictions = batch_predict(df)
                if predictions is not None:
                    display_batch_results(predictions, df)

        except Exception as e:
            st.error(f"Error reading CSV: {e}")

def random_sample_interface():
    """Random sample interface"""
    st.subheader("🎲 Generate Random Sample")

    if st.button("🎲 Generate Random Song", type="primary"):
        random_data = {}
        for feature in feature_names:
            random_data[feature] = np.random.normal(0, 1)

        st.subheader("🎵 Generated Song Features")
        df_display = pd.DataFrame(list(random_data.items()), columns=['Feature', 'Value'])
        df_display['Feature'] = df_display['Feature'].str.replace('_', ' ').str.title()
        df_display['Value'] = df_display['Value'].round(3)
        st.dataframe(df_display, use_container_width=True)

        make_prediction(random_data)

def make_prediction(input_data):
    """Make prediction and display results"""
    try:
        # Prepare data
        input_df = pd.DataFrame([input_data], columns=feature_names)
        input_df = input_df.fillna(0)

        # Scale data
        input_scaled = scaler.transform(input_df)

        # Make prediction
        prediction_prob = model.predict(input_scaled, verbose=0)[0][0]
        prediction_class = 1 if prediction_prob > 0.5 else 0
        confidence = max(prediction_prob, 1 - prediction_prob)

        # Display results
        display_prediction_results(prediction_prob, prediction_class, confidence)

    except Exception as e:
        st.error(f"Prediction error: {e}")

def display_prediction_results(prob, pred_class, confidence):
    """Display prediction results with fancy styling"""
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "🎯 Prediction",
            "High Popularity" if pred_class == 1 else "Low Popularity",
            delta=f"{confidence*100:.1f}% confidence"
        )

    with col2:
        st.metric(
            "📊 Probability",
            f"{prob*100:.1f}%",
            delta=f"{'↑' if prob > 0.5 else '↓'} {abs(prob-0.5)*200:.1f}%"
        )

    with col3:
        st.metric(
            "🎖️ Confidence",
            f"{confidence*100:.1f}%",
            delta="High" if confidence > 0.8 else "Medium" if confidence > 0.6 else "Low"
        )

    # Prediction card
    card_class = "high-popularity" if pred_class == 1 else "low-popularity"
    emoji = "🔥" if pred_class == 1 else "📉"

    st.markdown(f"""
    <div class="prediction-card {card_class}">
        <h2>{emoji} {'HIGH POPULARITY' if pred_class == 1 else 'LOW POPULARITY'}</h2>
        <h3>Probability: {prob*100:.1f}%</h3>
        <p>Confidence Level: {confidence*100:.1f}%</p>
    </div>
    """, unsafe_allow_html=True)

    # Probability gauge
    fig_gauge = go.Figure(go.Indicator(
        mode = "gauge+number+delta",
        value = prob * 100,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "Popularity Score"},
        delta = {'reference': 50},
        gauge = {
            'axis': {'range': [None, 100]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0, 50], 'color': "lightgray"},
                {'range': [50, 100], 'color': "gray"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 90
            }
        }
    ))
    fig_gauge.update_layout(height=400)
    st.plotly_chart(fig_gauge, use_container_width=True)

def create_feature_chart(input_data):
    """Create bar chart for features"""
    fig = px.bar(
        x=list(input_data.values()),
        y=list(input_data.keys()),
        orientation='h',
        title="Song Feature Values",
        color=list(input_data.values()),
        color_continuous_scale='viridis'
    )
    fig.update_layout(height=600)
    st.plotly_chart(fig, use_container_width=True)

def batch_predict(df):
    """Make predictions for batch of songs"""
    try:
        # Prepare data
        df_processed = df.copy()

        # Add missing columns with default values
        for feature in feature_names:
            if feature not in df_processed.columns:
                df_processed[feature] = 0

        # Reorder columns to match training
        df_processed = df_processed[feature_names]
        df_processed = df_processed.fillna(0)

        # Scale data
        scaled_data = scaler.transform(df_processed)

        # Make predictions
        predictions = model.predict(scaled_data, verbose=0)

        return predictions.flatten()

    except Exception as e:
        st.error(f"Batch prediction error: {e}")
        return None

def display_batch_results(predictions, original_df):
    """Display batch prediction results"""
    # Create results dataframe
    results_df = original_df.copy()
    results_df['Popularity_Probability'] = predictions
    results_df['Predicted_Class'] = (predictions > 0.5).astype(int)
    results_df['Prediction_Label'] = results_df['Predicted_Class'].map({
        1: 'High Popularity', 0: 'Low Popularity'
    })

    st.subheader("🎯 Batch Prediction Results")

    # Summary metrics
    col1, col2, col3 = st.columns(3)

    with col1:
        high_pop_count = (predictions > 0.5).sum()
        st.metric("🔥 High Popularity Songs", high_pop_count)

    with col2:
        avg_prob = predictions.mean()
        st.metric("📊 Average Probability", f"{avg_prob*100:.1f}%")

    with col3:
        max_prob = predictions.max()
        st.metric("⭐ Highest Score", f"{max_prob*100:.1f}%")

    # Results table
    st.dataframe(
        results_df[['Popularity_Probability', 'Predicted_Class', 'Prediction_Label']],
        use_container_width=True
    )

    # Distribution chart
    fig = px.histogram(
        x=predictions,
        nbins=20,
        title="Popularity Score Distribution",
        labels={'x': 'Popularity Probability', 'y': 'Count'}
    )
    st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    main()
