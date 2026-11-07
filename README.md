# 🍔 Food Delivery Time Predictor (ML Mini Project)

An end-to-end Machine Learning regression project that predicts food delivery arrival times in minutes, inspired by logistics architectures like **Zomato** and **Swiggy**.

Built with **Python**, **scikit-learn**, **Pandas**, **Seaborn**, and **Streamlit**.

---

## 📌 Problem Statement

In the food delivery industry, providing accurate Delivery Time Estimates (ETA) is essential for customer satisfaction, restaurant dispatch timing, and delivery fleet optimization. Delivery durations depend on numerous non-linear factors including:
- Geographical transit distance between the kitchen and customer drop point
- Courier profile (age, experience/ratings)
- Real-time road traffic congestion levels (Low, Medium, High, Jam)
- Weather impediments (Sunny, Fog, Sandstorms, Storms)
- Vehicle type and order preparation complexity

The goal of this project is to build, compare, and deploy regression models to accurately forecast `Time_taken(min)` based on these features.

---

## 📊 Dataset Description

The project utilizes the benchmark Food Delivery Dataset (Kaggle/Zomato logistics dataset), located in `data/train.csv` (and `archive/train.csv`).

Key attributes:
| Feature | Data Type | Description |
|---|---|---|
| `Delivery_person_Age` | Numerical | Age of the delivery partner (years) |
| `Delivery_person_Ratings` | Numerical | Partner service rating (1.0 to 5.0) |
| `Restaurant_latitude` & `longitude` | Coordinates | Geographic GPS location of restaurant |
| `Delivery_location_latitude` & `longitude` | Coordinates | Geographic GPS location of customer delivery destination |
| `Weatherconditions` | Categorical | Weather conditions (`Sunny`, `Cloudy`, `Windy`, `Fog`, `Sandstorms`, `Stormy`) |
| `Road_traffic_density` | Categorical | Current road traffic status (`Low`, `Medium`, `High`, `Jam`) |
| `Type_of_order` | Categorical | Food order type (`Snack`, `Meal`, `Drinks`, `Buffet`) |
| `Type_of_vehicle` | Categorical | Delivery vehicle mode (`motorcycle`, `scooter`, `electric_scooter`, `bicycle`) |
| `Time_taken(min)` | Target (Numerical) | Target variable: delivery duration in minutes |

### Data Cleaning & Preprocessing
1. **Whitespace & String Sanitization:** Stripped leading/trailing whitespaces across string features, removed `(min) ` prefix from target values, and removed `conditions ` prefix from weather categories.
2. **Missing & Invalid Values:** Filtered out `NaN` string artifacts, empty values, and handled missing fields.
3. **Geographic Coordinate Validation:** Inverted negative coordinate recording typos, removed invalid coordinate origins `(0, 0)`, and calculated great-circle distance.
4. **Feature Engineering:** Calculated **Haversine Distance (km)** using spherical trigonometry:
   $$\text{Haversine Distance} = 2 R \arcsin\left(\sqrt{\sin^2\left(\frac{\Delta \text{lat}}{2}\right) + \cos(\text{lat}_1)\cos(\text{lat}_2)\sin^2\left(\frac{\Delta \text{lon}}{2}\right)}\right)$$
   where $R = 6371.0 \text{ km}$. Raw coordinates and ID columns were dropped.
5. **Cleaned Dataset:** **40,109 high-quality rows** retained for modeling.

---

## 🛠️ Pipeline & Architecture

The machine learning workflow is implemented in [`src/train.py`](file:///c:/Users/rohit/Downloads/ALL%20projects/Miniproject2/src/train.py):

```
Raw CSV / Synthetic Data Generator
               │
               ▼
   Data Cleaning & Coordinate Sanitization
               │
               ▼
   Feature Engineering (Haversine Distance in km)
               │
               ▼
   scikit-learn ColumnTransformer:
     ├─ Numerical: StandardScaler (Age, Rating, Distance)
     └─ Categorical: OneHotEncoder (Weather, Traffic, Order, Vehicle)
               │
               ▼
   Model Training & Comparison (80/20 Split, random_state=42):
     ├─ Linear Regression
     ├─ Random Forest Regressor
     └─ Gradient Boosting Regressor
               │
               ▼
   Model Selection (Best by MAE) ──► models/best_model.joblib
   Metrics & Comparison Table    ──► outputs/model_comparison.csv
   Diagnostic Visualizations     ──► outputs/*.png
```

---

## 📈 Model Performance & Results

The models were evaluated on an independent 20% holdout test set (8,022 test samples).

| Model | MAE (Mean Absolute Error) | RMSE (Root Mean Squared Error) | R² Score |
|---|:---:|:---:|:---:|
| 🏆 **Random Forest Regressor** | **3.6810 min** | **4.6990 min** | **0.7449** |
| 🥈 **Gradient Boosting Regressor** | 3.6855 min | 4.6738 min | 0.7476 |
| 🥉 **Linear Regression** | 5.2556 min | 6.6492 min | 0.4891 |

- **Best Model:** **Random Forest Regressor** achieved the lowest Mean Absolute Error (**3.68 minutes** average error per order) and explains **~74.5% of variance** ($R^2 = 0.7449$).
- **Gradient Boosting** is an extremely close runner-up with an identical MAE of ~3.68 minutes and slightly lower RMSE (4.67 minutes).
- Non-linear tree ensemble architectures dramatically outperformed Linear Regression by capturing cross-feature interactions (e.g. adverse weather combined with heavy traffic jams).

### Diagnostic Plots Generated
All generated plots are saved to the [`outputs/`](file:///c:/Users/rohit/Downloads/ALL%20projects/Miniproject2/outputs) folder:
1. `outputs/actual_vs_predicted.png`: Scatter plot comparing ground truth delivery times against model predictions, demonstrating high linearity along the ideal $y=x$ line.
2. `outputs/residual_distribution.png`: Symmetric, zero-centered Gaussian residual distribution indicating unbiased predictions.
3. `outputs/feature_importance.png`: Bar chart displaying the top 10 most influential predictors.
4. `outputs/time_vs_distance.png`: Scatter plot and regression trend line illustrating how delivery time scales with transit distance.

---

## 💡 Key Predictors & Insights

From tree-based feature importance analysis, the top factors driving food delivery duration are:

1. **Delivery Partner Ratings (24.97%):** Experienced, top-rated riders navigate routes more efficiently and minimize delays during dispatch.
2. **Transit Distance (16.77%):** Physical distance in kilometers calculated via Haversine formula is a primary linear determinant of travel time.
3. **Road Traffic Conditions (Low/Jam) (~17.4% combined):** Low traffic accelerates delivery significantly, whereas Jam conditions introduce unavoidable street delays.
4. **Courier Age (13.57%):** Correlates with delivery tempo and familiarity with local sectors.
5. **Weather Conditions (Sunny/Fog/Stormy) (~20.7% combined):** Sunny conditions provide fastest turnaround, while Fog, Sandstorms, and Storms severely hamper transit velocity.

---

## 💻 How to Run

### 1. Prerequisites & Virtual Environment
Make sure Python 3.10+ is installed. Create and activate a virtual environment:

```bash
# Windows (PowerShell)
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# Linux / macOS
python3 -m venv .venv
source .venv/bin/activate
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run Training Pipeline
To clean data, train all 3 models, output the comparison table, save the best model, and produce diagnostic plots:
```bash
python src/train.py
```

### 4. Launch the Streamlit Web Application
To run the interactive delivery time estimation interface:
```bash
streamlit run app.py
```
Open `http://localhost:8501` in your browser.

---

## 🌐 Production Deployment

This project supports two production deployment options:

### Option A: Deploy to Netlify (Instant, Zero Server)
The repository includes a production-ready client web application (`index.html` + `app.js` + `netlify.toml`) with an embedded client-side decision tree inference engine:
1. Go to [Netlify](https://app.netlify.com/) and log in with your GitHub account.
2. Click **"Add new site"** &gt; **"Import an existing project"**.
3. Select **GitHub** and choose the repository `Mini_Project2`.
4. Leave the **Build command** empty and **Publish directory** as `.` (or root).
5. Click **"Deploy site"**. Netlify will launch your live web app in ~10 seconds with a public `https://<site-name>.netlify.app` URL.

### Option B: Deploy to Streamlit Community Cloud (Python Server)
To deploy the Python Streamlit app (`app.py`):
1. Go to [share.streamlit.io](https://share.streamlit.io/) and log in with GitHub.
2. Click **"New app"**, select `Mini_Project2`, branch `main`, and main file `app.py`.
3. Click **"Deploy!"** to get a live `https://<app-name>.streamlit.app` link.

---

## 📁 Project Structure

```
Mini_Project2/
├── index.html                 # Production Netlify Web Application
├── app.js                     # Client-side ML inference engine (Gradient Boosting)
├── netlify.toml               # Netlify build and security configuration
├── .gitignore                 # Production Git ignore rules
├── app.py                     # Streamlit Python Web Application
├── requirements.txt           # Project Python dependencies
├── README.md                  # Comprehensive project documentation
├── REPORT.md                  # Academic/College project report
├── data/
│   └── train.csv              # Cleaned food delivery dataset
├── models/
│   ├── best_model.joblib      # Saved scikit-learn best model pipeline (Random Forest)
│   └── model_metadata.joblib  # Feature definitions and model metadata
├── outputs/
│   ├── model_comparison.csv   # Benchmark table comparing LR, RF, GBR
│   ├── feature_importance_top10.csv # Top 10 feature importance scores
│   ├── actual_vs_predicted.png# Plot: Actual vs Predicted times
│   ├── residual_distribution.png # Plot: Error residual distribution
│   ├── feature_importance.png # Plot: Top 10 influential features
│   └── time_vs_distance.png   # Plot: Delivery time vs distance trend
└── src/
    └── train.py               # Complete data loading, preprocessing, and training pipeline
```
