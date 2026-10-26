"""
Food Delivery Time Predictor - Streamlit Web Application
Predicts delivery time in minutes using trained Machine Learning models (Swiggy / Zomato style).
"""

import os
import joblib
import pandas as pd
import streamlit as st

# Configure Streamlit page layout and appearance
st.set_page_config(
    page_title="Food Delivery Time Predictor",
    page_icon="🍔",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E293B;
        margin-bottom: 0.2rem;
    }
    .subtitle {
        font-size: 1.05rem;
        color: #64748B;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #F8FAFC 0%, #EFF6FF 100%);
        border: 1px solid #DBEAFE;
        border-radius: 12px;
        padding: 18px 22px;
        box-shadow: 0 2px 6px rgba(0, 0, 0, 0.04);
        margin-bottom: 15px;
    }
    .prediction-card {
        background: linear-gradient(135deg, #1E3A8A 0%, #2563EB 100%);
        color: white;
        border-radius: 14px;
        padding: 24px;
        text-align: center;
        box-shadow: 0 8px 20px rgba(37, 99, 235, 0.25);
        margin-top: 15px;
    }
    .prediction-value {
        font-size: 3.2rem;
        font-weight: 800;
        line-height: 1.1;
        margin: 10px 0;
    }
    .prediction-sub {
        font-size: 1.05rem;
        opacity: 0.9;
    }
    .badge-pill {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 9999px;
        font-size: 0.85rem;
        font-weight: 600;
        background-color: #E0E7FF;
        color: #3730A3;
        margin-right: 6px;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_model_and_meta():
    """
    Loads the trained pipeline model and metadata dictionary.
    """
    model_path = "models/best_model.joblib"
    meta_path = "models/model_metadata.joblib"

    if not os.path.exists(model_path):
        return None, None

    model = joblib.load(model_path)
    meta = joblib.load(meta_path) if os.path.exists(meta_path) else None
    return model, meta


@st.cache_data
def load_comparison():
    """
    Loads the model comparison metrics CSV.
    """
    comp_path = "outputs/model_comparison.csv"
    if os.path.exists(comp_path):
        return pd.read_csv(comp_path)
    return None


model, meta = load_model_and_meta()
comp_df = load_comparison()

# Sidebar Navigation & Quick Info
with st.sidebar:
    st.image("https://images.unsplash.com/photo-1526367790999-0150786686a2?auto=format&fit=crop&w=400&q=80", use_container_width=True)
    st.markdown("### 🍔 Delivery Predictor")
    st.markdown(
        "Instant ML-driven delivery time estimation inspired by logistics platforms like **Zomato** & **Swiggy**."
    )
    st.divider()

    if meta and 'metrics' in meta:
        st.markdown("#### 🏆 Active Model")
        st.markdown(f"**Model:** `{meta['best_model_name']}`")
        st.markdown(f"**MAE:** `{meta['metrics']['MAE']} min`")
        st.markdown(f"**RMSE:** `{meta['metrics']['RMSE']} min`")
        st.markdown(f"**R² Score:** `{meta['metrics']['R2']}`")
    else:
        st.info("Train the model via `python src/train.py` to see metrics.")

    st.divider()
    st.caption("Mini ML Project • Fast & Accurate Regression Engine")

# Main Header
st.markdown("<div class='main-title'>⚡ Food Delivery Time Predictor</div>", unsafe_allow_html=True)
st.markdown(
    "<div class='subtitle'>Predict food delivery arrival times based on distance, traffic conditions, weather, and courier profile.</div>",
    unsafe_allow_html=True
)

tab1, tab2, tab3 = st.tabs(["🚀 Predict Delivery Time", "📊 Model Comparison & Plots", "📖 About Project"])

# ----------------- TAB 1: PREDICTION -----------------
with tab1:
    if model is None:
        st.error("⚠️ Trained model not found! Please run `python src/train.py` to build and save the model.")
    else:
        # Pre-seed initial prediction for immediate display
        if 'last_prediction' not in st.session_state:
            default_df = pd.DataFrame([{
                'Delivery_person_Age': 28.0,
                'Delivery_person_Ratings': 4.7,
                'distance_km': 5.5,
                'Weatherconditions': 'Sunny',
                'Road_traffic_density': 'Medium',
                'Type_of_order': 'Meal',
                'Type_of_vehicle': 'motorcycle'
            }])
            default_time = max(10, round(float(model.predict(default_df)[0])))
            st.session_state['last_prediction'] = {
                'minutes': default_time,
                'window_low': max(10, default_time - 3),
                'window_high': default_time + 3,
                'distance': 5.5,
                'traffic': 'Medium',
                'weather': 'Sunny',
                'vehicle': 'motorcycle',
                'age': 28,
                'rating': 4.7,
                'order_type': 'Meal'
            }

        st.markdown("#### Enter Order & Delivery Details")

        with st.form("prediction_form"):
            col1, col2, col3 = st.columns(3)

            with col1:
                st.markdown("##### 🚴 Courier Information")
                age = st.slider(
                    "Courier Age (years)",
                    min_value=18,
                    max_value=50,
                    value=28,
                    help="Age of the delivery partner"
                )
                rating = st.slider(
                    "Courier Rating (1.0 to 5.0)",
                    min_value=1.0,
                    max_value=5.0,
                    value=4.7,
                    step=0.1,
                    help="Average rating of the delivery partner"
                )
                vehicle = st.selectbox(
                    "Type of Vehicle",
                    options=["motorcycle", "scooter", "electric_scooter", "bicycle"],
                    index=0,
                    help="Vehicle utilized for delivery"
                )

            with col2:
                st.markdown("##### 📍 Journey & Traffic")
                distance = st.number_input(
                    "Delivery Distance (km)",
                    min_value=0.2,
                    max_value=50.0,
                    value=5.5,
                    step=0.5,
                    help="Haversine distance between restaurant and destination in kilometers"
                )
                traffic = st.selectbox(
                    "Road Traffic Density",
                    options=["Low", "Medium", "High", "Jam"],
                    index=1,
                    help="Current road traffic level along the transit route"
                )
                order_type = st.selectbox(
                    "Type of Order",
                    options=["Snack", "Meal", "Drinks", "Buffet"],
                    index=1,
                    help="Category of food order being prepared"
                )

            with col3:
                st.markdown("##### 🌦️ Weather Conditions")
                weather = st.selectbox(
                    "Current Weather",
                    options=["Sunny", "Cloudy", "Windy", "Fog", "Sandstorms", "Stormy"],
                    index=0,
                    help="Prevailing weather during delivery"
                )

            st.markdown("<br>", unsafe_allow_html=True)
            predict_button = st.form_submit_button(
                "⚡ Predict Delivery Time",
                use_container_width=True,
                type="primary"
            )

        if predict_button:
            # Construct DataFrame exactly matching training features
            input_data = pd.DataFrame([{
                'Delivery_person_Age': float(age),
                'Delivery_person_Ratings': float(rating),
                'distance_km': float(distance),
                'Weatherconditions': weather,
                'Road_traffic_density': traffic,
                'Type_of_order': order_type,
                'Type_of_vehicle': vehicle
            }])

            with st.spinner("Calculating optimal delivery ETA..."):
                predicted_time = model.predict(input_data)[0]
                est_minutes = max(10, round(float(predicted_time)))
                window_low = max(10, est_minutes - 3)
                window_high = est_minutes + 3

            st.session_state['last_prediction'] = {
                'minutes': est_minutes,
                'window_low': window_low,
                'window_high': window_high,
                'distance': distance,
                'traffic': traffic,
                'weather': weather,
                'vehicle': vehicle,
                'age': age,
                'rating': rating,
                'order_type': order_type
            }

        if 'last_prediction' in st.session_state:
            pred_data = st.session_state['last_prediction']
            est_minutes = pred_data['minutes']
            window_low = pred_data['window_low']
            window_high = pred_data['window_high']

            st.divider()
            st.markdown("### 🎯 Prediction Results")
            res_col1, res_col2 = st.columns([1.2, 1])

            with res_col1:
                st.markdown(f"""
                <div class="prediction-card">
                    <div class="prediction-sub">Estimated Arrival Time</div>
                    <div class="prediction-value">{est_minutes} <span style="font-size: 1.8rem; font-weight: 500;">mins</span></div>
                    <div class="prediction-sub">Expected delivery window: <b>{window_low} – {window_high} minutes</b></div>
                </div>
                """, unsafe_allow_html=True)

            with res_col2:
                st.markdown("#### 📋 Order Summary")
                st.markdown(f"- **Distance:** `{pred_data['distance']} km`")
                st.markdown(f"- **Traffic Level:** `{pred_data['traffic']}`")
                st.markdown(f"- **Weather:** `{pred_data['weather']}`")
                st.markdown(f"- **Courier:** `{pred_data['vehicle'].title()}` (Age: {pred_data['age']}, Rating: {pred_data['rating']}⭐)")
                st.markdown(f"- **Order Category:** `{pred_data['order_type']}`")

                if est_minutes <= 25:
                    st.success("🟢 Fast Delivery expected! Conditions are optimal.")
                elif est_minutes <= 40:
                    st.info("🟡 Standard delivery window. Normal transit traffic.")
                else:
                    st.warning("🔴 Potential delay anticipated due to weather or traffic congestion.")

# ----------------- TAB 2: MODEL COMPARISON & PLOTS -----------------
with tab2:
    st.markdown("### 📊 Model Evaluation & Diagnostic Insights")

    if comp_df is not None:
        st.markdown("#### 📈 Benchmark Comparison Table")
        st.markdown(
            "Evaluation metrics computed across an independent **80/20 train-test split** using scikit-learn standard metrics."
        )

        styled_df = comp_df.copy()
        styled_df['MAE'] = styled_df['MAE'].map('{:.3f} min'.format)
        styled_df['RMSE'] = styled_df['RMSE'].map('{:.3f} min'.format)
        styled_df['R2'] = styled_df['R2'].map('{:.4f}'.format)

        st.dataframe(styled_df, use_container_width=True)

        # Highlight key cards
        best_name = comp_df.iloc[0]['Model']
        best_mae = comp_df.iloc[0]['MAE']
        best_r2 = comp_df.iloc[0]['R2']

        card_col1, card_col2, card_col3 = st.columns(3)
        with card_col1:
            st.metric("🏆 Best Algorithm", best_name)
        with card_col2:
            st.metric("🎯 Best MAE", f"{best_mae:.2f} mins", delta="-Lower is better", delta_color="inverse")
        with card_col3:
            st.metric("📈 R² Score (Variance Explained)", f"{best_r2:.3f}", delta="+Higher is better")
    else:
        st.warning("Model comparison data not found at `outputs/model_comparison.csv`.")

    st.divider()
    st.markdown("#### 🖼️ Model Diagnostic Plots")

    plot_paths = {
        "Actual vs Predicted Delivery Time": "outputs/actual_vs_predicted.png",
        "Residual Distribution": "outputs/residual_distribution.png",
        "Top 10 Feature Importances": "outputs/feature_importance.png",
        "Delivery Time vs Distance (km)": "outputs/time_vs_distance.png"
    }

    p_col1, p_col2 = st.columns(2)

    with p_col1:
        if os.path.exists(plot_paths["Actual vs Predicted Delivery Time"]):
            st.image(
                plot_paths["Actual vs Predicted Delivery Time"],
                caption="Actual vs Predicted Time (Ideal y = x dashed line)",
                use_container_width=True
            )
        else:
            st.info("Plot not found.")

        if os.path.exists(plot_paths["Top 10 Feature Importances"]):
            st.image(
                plot_paths["Top 10 Feature Importances"],
                caption="Top 10 Most Influential Predictors",
                use_container_width=True
            )
        else:
            st.info("Plot not found.")

    with p_col2:
        if os.path.exists(plot_paths["Residual Distribution"]):
            st.image(
                plot_paths["Residual Distribution"],
                caption="Residual Error Distribution (Centering near 0 indicates low bias)",
                use_container_width=True
            )
        else:
            st.info("Plot not found.")

        if os.path.exists(plot_paths["Delivery Time vs Distance (km)"]):
            st.image(
                plot_paths["Delivery Time vs Distance (km)"],
                caption="Transit Time Trend as Physical Distance Increases",
                use_container_width=True
            )
        else:
            st.info("Plot not found.")

# ----------------- TAB 3: ABOUT PROJECT -----------------
with tab3:
    st.markdown("### 📖 About This Machine Learning Mini Project")
    st.markdown("""
    #### 🎯 Objective
    Accurately forecast food order delivery time (in minutes) for on-demand logistics apps (such as Swiggy or Zomato) based on real-world transit features.

    #### ⚙️ Feature Engineering & Preprocessing
    1. **Haversine Distance**: Computes great-circle distance (km) between restaurant coordinates and customer drop coordinates.
    2. **Coordinate & Data Cleaning**: Cleans negative lat/long recording anomalies, strips trailing whitespaces, and removes `(min)` and `conditions ` prefixes.
    3. **ColumnTransformer Pipeline**: Standard scales numeric metrics (`age`, `rating`, `distance_km`) and one-hot encodes categorical factors (`traffic`, `weather`, `order`, `vehicle`) inside a reproducible scikit-learn Pipeline.

    #### 🚀 Models Compared
    - **Linear Regression**: Baseline regression model capturing direct linear relations.
    - **Gradient Boosting Regressor**: Sequentially optimized decision trees minimizing residual errors.
    - **Random Forest Regressor**: Ensemble of bagging decision trees capturing non-linear feature interactions.

    #### 💡 Key Findings
    Courier rating, journey distance, traffic conditions (e.g. Low vs Jam), courier age, and adverse weather conditions (Fog, Sandstorms, Storms) represent the dominant predictors influencing final delivery times.
    """)
