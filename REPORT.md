# Project Report: Food Delivery Time Predictor

**Course:** Machine Learning Mini Project  
**Domain:** Predictive Analytics & Logistics Optimization  
**Architecture:** Scikit-Learn Regression Pipeline & Streamlit Web Interface  

---

## Abstract

Accurate prediction of food delivery turnaround time is vital for hyper-local on-demand delivery services like Zomato and Swiggy. This project develops an end-to-end machine learning system capable of forecasting total delivery duration in minutes (`Time_taken(min)`). Using real-world delivery data comprising over 45,000 transaction records, we engineered spatial features via the Haversine great-circle distance formula, cleaned geographic coordinate anomalies, and structured a reproducible scikit-learn preprocessing and modeling pipeline. Three regression algorithms—Linear Regression (baseline), Random Forest Regressor, and Gradient Boosting Regressor—were benchmarked using an 80/20 train-test split. The **Random Forest Regressor** demonstrated superior predictive performance, achieving a **Mean Absolute Error (MAE) of 3.681 minutes**, a **Root Mean Squared Error (RMSE) of 4.699 minutes**, and an **$R^2$ score of 0.7449**. Feature importance analysis revealed that courier rating, journey distance, road traffic density, and courier age are the primary drivers of delivery time. The finalized model is deployed as an interactive Streamlit web application.

---

## 1. Introduction

With the exponential surge in on-demand e-commerce and hyper-local food dispatch services, accurate delivery duration estimations (Estimated Time of Arrival - ETA) directly impact customer satisfaction, rider dispatch scheduling, and kitchen order throughput. Underestimating delivery time leads to customer frustration and complaints, while overestimating causes loss of order volume. 

Predicting delivery time is inherently challenging due to dynamic external variables such as traffic congestion patterns, unpredictable weather conditions, physical route distance, order preparation nuances, and courier efficiency. The primary objective of this project is to construct a robust regression pipeline to model these non-linear relationships and deliver real-time predictions via an accessible web application.

---

## 2. Dataset & Data Cleaning

The project uses the benchmark food delivery dataset comprising courier demographics, GPS coordinates, environmental conditions, and transit times.

### Key Features
1. **Courier Attributes:** `Delivery_person_Age`, `Delivery_person_Ratings`
2. **Spatial Coordinates:** `Restaurant_latitude`, `Restaurant_longitude`, `Delivery_location_latitude`, `Delivery_location_longitude`
3. **Environmental & Contextual Factors:** `Weatherconditions` (Sunny, Cloudy, Windy, Fog, Sandstorms, Stormy), `Road_traffic_density` (Low, Medium, High, Jam)
4. **Order & Transport Factors:** `Type_of_order` (Snack, Meal, Drinks, Buffet), `Type_of_vehicle` (motorcycle, scooter, electric_scooter, bicycle)
5. **Target Variable:** `Time_taken(min)` (integer minutes)

### Data Preprocessing
- **Text & Prefix Sanitization:** Stripped whitespace artifacts, removed `conditions ` prefix from weather conditions, and stripped `(min)` from the target variable.
- **Handling Nulls & Anomalies:** Converted numeric fields from string representations, replaced string-coded `'NaN'` with true null values, and filtered records with missing values.
- **Coordinate Correction:** Detected negative GPS coordinate records caused by data logging typos (India spans Northern and Eastern hemispheres) and took absolute coordinate values. Removed invalid `(0, 0)` coordinates.
- **Data Retained:** **40,109 clean, verified records** retained from the initial 45,593 rows.

---

## 3. Methodology & System Architecture

```
[Raw Food Delivery Data]
          │
          ▼
[Data Cleaning & Coordinate Typo Correction]
          │
          ▼
[Spatial Feature Engineering: Haversine Distance (km)]
          │
          ▼
[scikit-learn ColumnTransformer]
   ├─ Numeric Pipeline: StandardScaler (Age, Rating, Distance)
   └─ Categorical Pipeline: OneHotEncoder (Weather, Traffic, Order, Vehicle)
          │
          ▼
[80/20 Train-Test Split (32,087 Train / 8,022 Test, random_state=42)]
          │
          ▼
[Model Training & Comparison: Linear Regression, Random Forest, Gradient Boosting]
          │
          ▼
[Best Model Serialization (joblib) & Diagnostic Plots]
          │
          ▼
[Interactive Streamlit Application Deployment]
```

### 3.1 Feature Engineering: Haversine Distance
Physical road transit distance is derived from geographical coordinates using the spherical Haversine formula:
$$d = 2 R \arcsin\left(\sqrt{\sin^2\left(\frac{\Delta \phi}{2}\right) + \cos(\phi_1)\cos(\phi_2)\sin^2\left(\frac{\Delta \lambda}{2}\right)}\right)$$
where $R = 6371 \text{ km}$, $\phi_1, \phi_2$ denote restaurant and delivery latitudes in radians, and $\Delta \lambda$ is the longitude difference. The raw GPS coordinates and transactional IDs were subsequently discarded to avoid spatial overfitting.

### 3.2 Preprocessing Pipeline
To guarantee reproducibility and avoid data leakage, preprocessing is encapsulated in a scikit-learn `ColumnTransformer`:
- **Numeric Features (`Delivery_person_Age`, `Delivery_person_Ratings`, `distance_km`):** Scaled via `StandardScaler`.
- **Categorical Features (`Weatherconditions`, `Road_traffic_density`, `Type_of_order`, `Type_of_vehicle`):** Encoded via `OneHotEncoder(handle_unknown='ignore')`.

### 3.3 Models Evaluated
1. **Linear Regression:** Baseline parametric estimator modeling direct linear relationships.
2. **Random Forest Regressor:** Bagging ensemble of 100 decision trees (`max_depth=16`, `min_samples_split=6`) to capture non-linear feature splits without severe overfitting.
3. **Gradient Boosting Regressor:** Sequential boosting ensemble of 120 estimators (`learning_rate=0.08`, `max_depth=5`) optimizing mean squared error residuals.

---

## 4. Experimental Results & Discussion

### 4.1 Quantitative Performance Comparison
Models were evaluated on the independent holdout test set ($N = 8,022$ samples) using Mean Absolute Error (MAE), Root Mean Squared Error (RMSE), and the Coefficient of Determination ($R^2$):

| Rank | Model Architecture | MAE (minutes) | RMSE (minutes) | $R^2$ Score |
|:---:|---|:---:|:---:|:---:|
| 🥇 | **Random Forest Regressor** | **3.6810** | **4.6990** | **0.7449** |
| 🥈 | **Gradient Boosting Regressor** | **3.6855** | **4.6738** | **0.7476** |
| 🥉 | **Linear Regression** | 5.2556 | 6.6492 | 0.4891 |

### 4.2 Diagnostic Findings
- **Ensemble Superiority:** Tree-based ensembles achieved an average error reduction of **~1.57 minutes** per delivery compared to Linear Regression ($MAE = 3.68$ min vs $5.26$ min), boosting variance explained from $48.9\%$ to **$74.5\%$**.
- **Residual Analysis:** The residual error distribution ($y - \hat{y}$) for the Random Forest model exhibits a sharp, zero-centered Gaussian profile with minimal skewness, confirming low model bias.
- **Actual vs Predicted Fit:** Predictions align tightly around the ideal $y = x$ reference axis across the full 15–50 minute spectrum.

### 4.3 Feature Importance Ranking (Top Predictors)
Extracting MDI (Mean Decrease in Impurity) from the best Random Forest model highlights the most decisive factors:
1. **Courier Rating (`Delivery_person_Ratings`) — 24.97%:** Experienced couriers navigate faster, have lower pickup delays, and optimize delivery handoffs.
2. **Haversine Distance (`distance_km`) — 16.77%:** Strong baseline determinant of physical transit duration.
3. **Low Road Traffic (`Road_traffic_density: Low`) — 14.41%:** Open roads allow unimpeded vehicle transit velocity.
4. **Courier Age (`Delivery_person_Age`) — 13.57%:** Significant demographic indicator linked to delivery experience.
5. **Weather Conditions (`Sunny: 8.51%`, `Fog: 6.25%`, `Cloudy: 5.96%`):** Adverse weather creates severe systemic slowdowns, while clear sunny skies expedite arrivals.
6. **Vehicle Mode (`motorcycle`) — 3.45%:** Agile two-wheelers navigate urban congestion faster than larger vehicles.

---

## 5. Web Application & Deployment

The trained model pipeline is serialized as a standalone joblib artifact (`models/best_model.joblib`) and integrated into an interactive **Streamlit** web application (`app.py`):
- **Tab 1: Predict Delivery Time:** Real-time user input form (rider age, rating, distance, traffic, weather, order type, vehicle) outputting instant ETA in minutes with an arrival confidence window.
- **Tab 2: Model Comparison & Diagnostics:** Interactive display of the benchmark performance table and the four high-resolution diagnostic plots (`actual_vs_predicted.png`, `residual_distribution.png`, `feature_importance.png`, `time_vs_distance.png`).

---

## 6. Conclusion & Future Scope

### Conclusion
This project successfully designed, implemented, and deployed a complete Machine Learning solution for food delivery duration prediction. The **Random Forest Regressor** proved to be the most accurate model, estimating delivery times within an average error margin of **±3.68 minutes** and accounting for **74.5% of variance**. The results confirm that courier performance ratings and physical journey distance, coupled with traffic and weather impediments, dictate the vast majority of delivery duration dynamics.

### Future Scope
1. **Live GPS Routing & Road Distance:** Replacing Euclidean/Haversine distance with actual OpenStreetMap (OSRM) or Google Maps Directions API routing to account for road curvature and one-way streets.
2. **Deep Learning & Temporal Features:** Incorporating LSTM or Transformer models to track time-of-day rush hour spikes and historical restaurant prep delays.
3. **Real-Time Fleet Dispatch:** Integrating the model into active multi-stop dispatch algorithms to optimize batch delivery allocation across clusters of orders.

---

## References
1. Breiman, L. (2001). *Random Forests*. Machine Learning, 45(1), 5-32.
2. Pedregosa, F., et al. (2011). *Scikit-learn: Machine Learning in Python*. JMLR, 12, 2825-2830.
3. Sinnott, R. W. (1984). *Virtues of the Haversine*. Sky and Telescope, 68(2), 159.
