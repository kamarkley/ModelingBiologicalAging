import pandas as pd
import numpy as np
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import ElasticNet
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import LinearSVR
from sklearn.inspection import permutation_importance

### Step 1 --> Load Data
# assume NHANES_SUPER.csv is in curr dir
df = pd.read_csv('NHANES_SUPER.csv')

# define features and target
clinical_features = [
    'LBXGH',  # HbA1c
    'LBXTC',  # Total Cholesterol
    'LBXCRP',  # C-Reactive Protein
    'BPXSY1',  # Systolic BP
    'BMXBMI',  # BMI
    'BMXWAIST',  # Waist Circumference
    'SMQ020'  # Smoking Status (Categorical)
]

demographic_features = [
    'RIDAGEYR',  # Age
    'RIAGENDR',  # Gender (Categorical)
    'RIDRETH1',  # Race (Categorical)
    'INDFMPIR',  # Income-to-Poverty Ratio (Numeric)
    'DMDEDUC2',  # Education (Categorical)
    'DMDMARTL',  # Marital Status (Categorical)
    'DMDHHSIZ'  # Household Size (Numeric)
]
target = 'TELOMEAN'

### STEP 2 --> Data Cleaning Protocol

print(f"Initial cohort size: {len(df)}")

# define specific sentinel non-biological/survey codes to remove
sentinels = [888, 8888, 7.0, 9.0, 7, 9]

# replace sentinels with NaN ONLY in our target features to avoid wiping valid data elsewhere
features_to_clean = clinical_features + demographic_features
df[features_to_clean] = df[features_to_clean].replace(sentinels, np.nan)

# filter for complete cases: drop rows where any feature OR target is now NaN
modeling_columns = features_to_clean + [target]
df_clean = df.dropna(subset=modeling_columns)

print(f"Cleaned cohort size (complete cases): {len(df_clean)}")

### STEP 3 --> Setup X and y
X = df_clean[features_to_clean]
y = df_clean[target]

### STEP 4 --> Preprocessing Pipeline
numeric_transformer = Pipeline(steps=[('scaler', StandardScaler())])
categorical_transformer = Pipeline(steps=[('onehot', OneHotEncoder(handle_unknown='ignore'))])

# carefully map new features to correct mathematical type
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer,
         ['RIDAGEYR', 'INDFMPIR', 'DMDHHSIZ', 'BMXBMI', 'BMXWAIST', 'BPXSY1', 'LBXTC', 'LBXGH', 'LBXCRP']),
        ('cat', categorical_transformer, ['RIAGENDR', 'RIDRETH1', 'DMDEDUC2', 'DMDMARTL', 'SMQ020'])
    ])

### STEP 5 --> Pipelines & Parameter Grids
# Initialize base models (parameters will be tuned via GridSearch)
en_pipe = Pipeline(steps=[('preprocessor', preprocessor),
                          ('regressor', ElasticNet(random_state=42))])

svr_pipe = Pipeline(steps=[('preprocessor', preprocessor),
                           ('regressor', LinearSVR(random_state=42, dual='auto', max_iter=250000))])

rf_pipe = Pipeline(steps=[('preprocessor', preprocessor),
                          ('regressor', RandomForestRegressor(random_state=42, n_jobs=-1))])

# Define Search Spaces (Note: 'regressor__' prefix targets the regressor step in the pipeline)
en_grid = {
    'regressor__alpha': [0.0001, 0.001, 0.01, 0.1, 1.0],
    'regressor__l1_ratio': [0.1, 0.5, 0.7, 0.9, 1.0]
}

svr_grid = {
    'regressor__C': [0.1, 1.0, 10.0],
    'regressor__epsilon': [0.01, 0.1, 1.0]
}

rf_grid = {
    'regressor__n_estimators': [100, 300, 500],
    'regressor__max_depth': [None, 10, 20]
}

### STEP 6 & 7 --> K-Fold GridSearchCV & Evaluation
# Setup K-Fold
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Tuple structure: (display name, pipeline, parameter grid)
models = [
    ("Elastic Net", en_pipe, en_grid),
    ("Linear SVR", svr_pipe, svr_grid),
    ("Random Forest", rf_pipe, rf_grid)
]

# Track metrics for final table
performance_metrics = []

# Define scoring dictionary to calculate multiple metrics simultaneously
scoring = {
    'r2': 'r2',
    'neg_rmse': 'neg_root_mean_squared_error'
}

for name, pipe, param_grid in models:
    print(f"\nRunning GridSearchCV for {name}...")

    # 1. Run Grid Search
    # refit='r2' tells the search to optimize for the highest R2 score
    grid_search = GridSearchCV(pipe, param_grid, cv=kf, scoring=scoring, refit='r2', n_jobs=-1)
    grid_search.fit(X, y)

    # 2. Extract best model and its performance
    best_model = grid_search.best_estimator_
    best_idx = grid_search.best_index_

    mean_r2 = grid_search.cv_results_['mean_test_r2'][best_idx]
    mean_rmse = -grid_search.cv_results_['mean_test_neg_rmse'][best_idx]  # Flip sign

    # Clean up the parameters dictionary for printing (remove 'regressor__')
    clean_params = ", ".join([f"{k.replace('regressor__', '')}: {v}" for k, v in grid_search.best_params_.items()])

    # 3. Append to metrics list
    performance_metrics.append({
        "Model Architecture": name,
        "CV Mean R2": mean_r2,
        "CV Mean RMSE": mean_rmse,
        "Optimal Parameters": clean_params
    })

    # 4. Feature Importance
    # Calculate permutation importance using the objectively best version of the model
    print(f"Calculating Permutation Importance for best {name}...")
    result = permutation_importance(best_model, X, y, n_repeats=10, random_state=42, n_jobs=-1)

    print(f"\n--- {name} Top Drivers ---")
    sorted_idx = result.importances_mean.argsort()[::-1]

    # Print the top features
    for i in sorted_idx:
        feature_name = X.columns[i]
        importance_mean = result.importances_mean[i]
        importance_std = result.importances_std[i]
        print(f"{feature_name}: {importance_mean:.4f} +/- {importance_std:.4f}")

### STEP 8 --> Print the Summary Table
print("\n" + "=" * 90)
print("Table 1: 5-Fold Cross-Validation & GridSearch Optimization Results")
print("=" * 90)

# convert to df for clean printing
results_df = pd.DataFrame(performance_metrics)

# format float to 4 decimal places
results_df['CV Mean R2'] = results_df['CV Mean R2'].apply(lambda x: f"{x:.4f}")
results_df['CV Mean RMSE'] = results_df['CV Mean RMSE'].apply(lambda x: f"{x:.4f}")

# print without index col for clean look
print(results_df.to_string(index=False))
print("=" * 90 + "\n")
