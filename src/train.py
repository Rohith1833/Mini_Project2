"""
Food Delivery Time Predictor - Training Pipeline
Trains and evaluates Linear Regression, Random Forest, and Gradient Boosting models.
"""

import os
import glob
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score


def calculate_haversine(lat1, lon1, lat2, lon2):
    """
    Calculate the great circle distance between two points on earth in kilometers.
    """
    R = 6371.0  # Earth radius in kilometers
    phi1 = np.radians(lat1)
    phi2 = np.radians(lat2)
    delta_phi = np.radians(lat2 - lat1)
    delta_lambda = np.radians(lon2 - lon1)

    a = (np.sin(delta_phi / 2.0) ** 2 +
         np.cos(phi1) * np.cos(phi2) * np.sin(delta_lambda / 2.0) ** 2)
    c = 2 * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))
    return R * c


def generate_synthetic_data(filepath="data/train.csv", n_rows=20000):
    """
    Generates a realistic synthetic food delivery dataset if no CSV is found.
    """
    print(f"Generating realistic synthetic dataset ({n_rows} rows)...")
    np.random.seed(42)

    ages = np.random.randint(19, 40, size=n_rows)
    ratings = np.round(np.random.uniform(3.5, 5.0, size=n_rows), 1)

    # Coordinates centered around Bangalore metropolitan area
    base_lat = 12.9716
    base_lon = 77.5946
    rest_lat = base_lat + np.random.normal(0, 0.05, size=n_rows)
    rest_lon = base_lon + np.random.normal(0, 0.05, size=n_rows)
    del_lat = rest_lat + np.random.normal(0, 0.06, size=n_rows)
    del_lon = rest_lon + np.random.normal(0, 0.06, size=n_rows)

    weather_choices = ['Sunny', 'Stormy', 'Sandstorms', 'Cloudy', 'Fog', 'Windy']
    traffic_choices = ['Low', 'Medium', 'High', 'Jam']
    order_choices = ['Snack', 'Meal', 'Drinks', 'Buffet']
    vehicle_choices = ['motorcycle', 'scooter', 'electric_scooter', 'bicycle']

    weather = np.random.choice(weather_choices, size=n_rows, p=[0.25, 0.15, 0.10, 0.20, 0.15, 0.15])
    traffic = np.random.choice(traffic_choices, size=n_rows, p=[0.25, 0.30, 0.25, 0.20])
    order = np.random.choice(order_choices, size=n_rows)
    vehicle = np.random.choice(vehicle_choices, size=n_rows, p=[0.45, 0.35, 0.15, 0.05])

    dist = calculate_haversine(rest_lat, rest_lon, del_lat, del_lon)

    traffic_penalties = {'Low': 0, 'Medium': 5, 'High': 11, 'Jam': 19}
    weather_penalties = {'Sunny': 0, 'Cloudy': 2, 'Windy': 4, 'Sandstorms': 7, 'Fog': 8, 'Stormy': 10}
    vehicle_factors = {'motorcycle': 0, 'scooter': 1, 'electric_scooter': 2, 'bicycle': 7}

    traffic_cost = np.array([traffic_penalties[t] for t in traffic])
    weather_cost = np.array([weather_penalties[w] for w in weather])
    vehicle_cost = np.array([vehicle_factors[v] for v in vehicle])
    rating_bonus = (ratings - 4.0) * -2.5

    base_prep_time = 12.0
    travel_time = dist * 1.5
    noise = np.random.normal(0, 3.5, size=n_rows)

    delivery_time = base_prep_time + travel_time + traffic_cost + weather_cost + vehicle_cost + rating_bonus + noise
    delivery_time = np.clip(np.round(delivery_time), 10, 80).astype(int)

    df_synth = pd.DataFrame({
        'ID': [f'0x{i:04x}' for i in range(n_rows)],
        'Delivery_person_ID': [f'SYNTHRES{i%50:02d}DEL01' for i in range(n_rows)],
        'Delivery_person_Age': ages,
        'Delivery_person_Ratings': ratings,
        'Restaurant_latitude': rest_lat,
        'Restaurant_longitude': rest_lon,
        'Delivery_location_latitude': del_lat,
        'Delivery_location_longitude': del_lon,
        'Weatherconditions': [f'conditions {w}' for w in weather],
        'Road_traffic_density': [f'{t} ' for t in traffic],
        'Type_of_order': [f'{o} ' for o in order],
        'Type_of_vehicle': [f'{v} ' for v in vehicle],
        'Time_taken(min)': [f'(min) {t}' for t in delivery_time]
    })

    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    df_synth.to_csv(filepath, index=False)
    print(f"Synthetic dataset saved to {filepath} ({len(df_synth)} rows).")
    return df_synth


def find_dataset():
    """
    Search for train.csv or any relevant CSV in the project folder.
    """
    candidates = [
        "data/train.csv",
        "archive/train.csv",
        "train.csv",
    ]
    for c in candidates:
        if os.path.exists(c):
            print(f"Found dataset at: {c}")
            return c

    all_csvs = glob.glob("**/*.csv", recursive=True)
    for c in all_csvs:
        if "train" in os.path.basename(c).lower():
            print(f"Found candidate dataset at: {c}")
            return c

    return None


def load_and_clean_data():
    """
    Loads raw CSV, strips whitespace/'NaN', cleans columns, filters invalid coordinates,
    computes haversine distance, and drops unnecessary columns.
    """
    csv_path = find_dataset()
    if csv_path is None:
        df_raw = generate_synthetic_data()
    else:
        print(f"Loading data from {csv_path}...")
        df_raw = pd.read_csv(csv_path)

    expected_cols = [
        'Delivery_person_Age', 'Delivery_person_Ratings',
        'Restaurant_latitude', 'Restaurant_longitude',
        'Delivery_location_latitude', 'Delivery_location_longitude',
        'Weatherconditions', 'Road_traffic_density',
        'Type_of_order', 'Type_of_vehicle', 'Time_taken(min)'
    ]

    missing = [c for c in expected_cols if c not in df_raw.columns]
    if missing:
        raise ValueError(f"Required columns missing from dataset: {missing}")

    df = df_raw[expected_cols].copy()

    # 1. Strip whitespace on string/categorical columns
    for col in df.columns:
        if not pd.api.types.is_numeric_dtype(df[col]):
            df[col] = df[col].astype(str).str.strip()

    # 2. Replace 'NaN' and empty strings with np.nan
    df.replace(['NaN', 'conditions NaN', 'nan', 'null', ''], np.nan, inplace=True)

    # 3. Clean 'conditions ' prefix from Weatherconditions
    df['Weatherconditions'] = df['Weatherconditions'].astype(str).str.replace('conditions ', '', regex=False).str.strip()

    # 4. Clean '(min) ' prefix from target Time_taken(min)
    df['Time_taken(min)'] = df['Time_taken(min)'].astype(str).str.replace('(min)', '', regex=False).str.strip()

    # 5. Convert numeric columns
    numeric_cols = [
        'Delivery_person_Age', 'Delivery_person_Ratings',
        'Restaurant_latitude', 'Restaurant_longitude',
        'Delivery_location_latitude', 'Delivery_location_longitude',
        'Time_taken(min)'
    ]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # Drop nulls across essential fields
    initial_count = len(df)
    df = df.dropna(subset=['Delivery_person_Age', 'Delivery_person_Ratings',
                          'Weatherconditions', 'Road_traffic_density',
                          'Type_of_order', 'Type_of_vehicle', 'Time_taken(min)'])

    # 6. Coordinate validation and distance computation
    # In Indian delivery datasets, negative lat/lon are recording typos (signs reversed)
    lat1 = df['Restaurant_latitude'].abs()
    lon1 = df['Restaurant_longitude'].abs()
    lat2 = df['Delivery_location_latitude'].abs()
    lon2 = df['Delivery_location_longitude'].abs()

    # Valid non-zero coordinate mask within valid geographic bounds
    valid_coords = (
        (lat1 > 0.0) & (lon1 > 0.0) & (lat2 > 0.0) & (lon2 > 0.0) &
        (lat1 <= 90.0) & (lon1 <= 180.0) & (lat2 <= 90.0) & (lon2 <= 180.0)
    )
    df = df[valid_coords].copy()

    # Feature Engineering: Haversine distance in km
    dist = calculate_haversine(lat1[valid_coords], lon1[valid_coords], lat2[valid_coords], lon2[valid_coords])
    df['distance_km'] = np.round(dist, 2)

    # Filter out anomalous distances (> 60 km for city food delivery or < 0.1 km)
    df = df[(df['distance_km'] >= 0.1) & (df['distance_km'] <= 60.0)].copy()

    # Drop raw coordinate columns
    drop_cols = ['Restaurant_latitude', 'Restaurant_longitude',
                 'Delivery_location_latitude', 'Delivery_location_longitude']
    df = df.drop(columns=drop_cols)

    print(f"Data cleaning finished: {len(df)} rows retained (from {initial_count} raw rows).")
    return df


def train_and_evaluate():
    """
    Runs model training, comparison, artifact saving, and visualization generation.
    """
    os.makedirs("models", exist_ok=True)
    os.makedirs("outputs", exist_ok=True)

    df = load_and_clean_data()

    numeric_features = ['Delivery_person_Age', 'Delivery_person_Ratings', 'distance_km']
    categorical_features = ['Weatherconditions', 'Road_traffic_density', 'Type_of_order', 'Type_of_vehicle']
    target = 'Time_taken(min)'

    X = df[numeric_features + categorical_features]
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42
    )

    print(f"Training set: {X_train.shape[0]} rows, Test set: {X_test.shape[0]} rows")

    # Scikit-learn ColumnTransformer
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features)
        ]
    )

    models = {
        'Linear Regression': LinearRegression(),
        'Random Forest': RandomForestRegressor(n_estimators=100, max_depth=16, min_samples_split=6, random_state=42, n_jobs=-1),
        'Gradient Boosting': GradientBoostingRegressor(n_estimators=120, learning_rate=0.08, max_depth=5, random_state=42)
    }

    results = []
    fitted_pipelines = {}
    predictions = {}

    for name, regressor in models.items():
        print(f"Training {name}...")
        pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('regressor', regressor)
        ])

        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)

        mae = mean_absolute_error(y_test, y_pred)
        rmse = root_mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        results.append({
            'Model': name,
            'MAE': round(mae, 4),
            'RMSE': round(rmse, 4),
            'R2': round(r2, 4)
        })

        fitted_pipelines[name] = pipeline
        predictions[name] = y_pred
        print(f"  -> {name} | MAE: {mae:.3f} min | RMSE: {rmse:.3f} min | R²: {r2:.4f}")

    # Model comparison table
    comparison_df = pd.DataFrame(results)
    comparison_df = comparison_df.sort_values(by='MAE').reset_index(drop=True)

    comparison_path = "outputs/model_comparison.csv"
    comparison_df.to_csv(comparison_path, index=False)
    print("\n=== Model Comparison Table ===")
    print(comparison_df.to_string(index=False))
    print(f"Saved comparison to: {comparison_path}")

    # Best model selection by MAE
    best_row = comparison_df.iloc[0]
    best_model_name = best_row['Model']
    best_pipeline = fitted_pipelines[best_model_name]
    best_y_pred = predictions[best_model_name]

    model_save_path = "models/best_model.joblib"
    joblib.dump(best_pipeline, model_save_path)
    print(f"\nBest Model: '{best_model_name}' (MAE: {best_row['MAE']} min, R²: {best_row['R2']})")
    print(f"Saved best model to: {model_save_path}")

    # Save metadata dictionary for App usage
    meta = {
        'best_model_name': best_model_name,
        'metrics': best_row.to_dict(),
        'numeric_features': numeric_features,
        'categorical_features': categorical_features,
        'categories': {
            col: sorted(list(df[col].dropna().unique())) for col in categorical_features
        },
        'feature_ranges': {
            'Delivery_person_Age': {'min': float(df['Delivery_person_Age'].min()), 'max': float(df['Delivery_person_Age'].max()), 'default': 28.0},
            'Delivery_person_Ratings': {'min': float(df['Delivery_person_Ratings'].min()), 'max': float(df['Delivery_person_Ratings'].max()), 'default': 4.7},
            'distance_km': {'min': float(df['distance_km'].min()), 'max': float(df['distance_km'].max()), 'default': 8.5}
        }
    }
    joblib.dump(meta, "models/model_metadata.joblib")

    # Generate Visualization Plots
    generate_plots(X_test, y_test, best_y_pred, best_pipeline, best_model_name, df)

    return comparison_df, best_row


def generate_plots(X_test, y_test, y_pred, pipeline, model_name, full_df):
    """
    Generates and saves the 4 required analytical plots to outputs/.
    """
    print("\nGenerating diagnostic plots in outputs/...")
    sns.set_theme(style="whitegrid", palette="muted")
    plt.rcParams.update({'font.sans-serif': 'DejaVu Sans', 'font.size': 11})

    # Plot 1: Actual vs Predicted Scatter
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(y_test, y_pred, alpha=0.35, color='#2563EB', edgecolors='none', s=24)
    min_val = min(y_test.min(), y_pred.min())
    max_val = max(y_test.max(), y_pred.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Ideal Fit (y = x)')
    ax.set_title(f'Actual vs Predicted Delivery Time ({model_name})', fontsize=13, fontweight='bold', pad=12)
    ax.set_xlabel('Actual Delivery Time (minutes)', fontsize=11)
    ax.set_ylabel('Predicted Delivery Time (minutes)', fontsize=11)
    ax.legend(frameon=True)
    plt.tight_layout()
    plot1_path = "outputs/actual_vs_predicted.png"
    plt.savefig(plot1_path, dpi=300)
    plt.close()
    print(f"Saved: {plot1_path}")

    # Plot 2: Residual Distribution
    residuals = y_test - y_pred
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.histplot(residuals, kde=True, color='#059669', bins=35, ax=ax, edgecolor='white', alpha=0.6)
    ax.axvline(0, color='#DC2626', linestyle='--', lw=2, label='Zero Error')
    ax.set_title(f'Residual Distribution ({model_name})', fontsize=13, fontweight='bold', pad=12)
    ax.set_xlabel('Residual (Actual - Predicted) [minutes]', fontsize=11)
    ax.set_ylabel('Count', fontsize=11)
    ax.legend(frameon=True)
    plt.tight_layout()
    plot2_path = "outputs/residual_distribution.png"
    plt.savefig(plot2_path, dpi=300)
    plt.close()
    print(f"Saved: {plot2_path}")

    # Plot 3: Feature Importance Bar Chart (Top 10)
    regressor = pipeline.named_steps['regressor']
    prep = pipeline.named_steps['preprocessor']

    feature_names = prep.get_feature_names_out()
    # Clean feature names for clean presentation (remove num__ and cat__ prefixes)
    cleaned_feature_names = [
        f.replace('num__', '').replace('cat__', '').replace('_', ' ')
        for f in feature_names
    ]

    importances = None
    if hasattr(regressor, 'feature_importances_'):
        importances = regressor.feature_importances_
    elif hasattr(regressor, 'coef_'):
        importances = np.abs(regressor.coef_)

    if importances is not None:
        fi_df = pd.DataFrame({
            'Feature': cleaned_feature_names,
            'Importance': importances
        }).sort_values(by='Importance', ascending=False).head(10)

        fig, ax = plt.subplots(figsize=(8, 6))
        palette = sns.color_palette("mako", n_colors=10)
        sns.barplot(data=fi_df, x='Importance', y='Feature', hue='Feature', legend=False, ax=ax, palette=palette)
        ax.set_title(f'Top 10 Feature Importances ({model_name})', fontsize=13, fontweight='bold', pad=12)
        ax.set_xlabel('Relative Importance Score', fontsize=11)
        ax.set_ylabel('Feature Name', fontsize=11)
        plt.tight_layout()
        plot3_path = "outputs/feature_importance.png"
        plt.savefig(plot3_path, dpi=300)
        plt.close()
        print(f"Saved: {plot3_path}")
        # Save top feature importances to CSV as well
        fi_df.to_csv("outputs/feature_importance_top10.csv", index=False)

    # Plot 4: Delivery Time vs Distance
    fig, ax = plt.subplots(figsize=(8, 6))
    sample_df = full_df.sample(n=min(3000, len(full_df)), random_state=42)
    sns.regplot(
        data=sample_df,
        x='distance_km',
        y='Time_taken(min)',
        scatter_kws={'alpha': 0.25, 'color': '#7C3AED', 's': 20},
        line_kws={'color': '#EA580C', 'lw': 2.5, 'label': 'Trend Line'},
        ax=ax
    )
    ax.set_title('Delivery Time vs Distance (km)', fontsize=13, fontweight='bold', pad=12)
    ax.set_xlabel('Haversine Distance (km)', fontsize=11)
    ax.set_ylabel('Time Taken (minutes)', fontsize=11)
    ax.legend(frameon=True)
    plt.tight_layout()
    plot4_path = "outputs/time_vs_distance.png"
    plt.savefig(plot4_path, dpi=300)
    plt.close()
    print(f"Saved: {plot4_path}")


if __name__ == "__main__":
    train_and_evaluate()
