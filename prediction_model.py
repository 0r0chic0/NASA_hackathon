import os
import numpy as np
import pandas as pd
from obspy import read
from obspy.signal.trigger import classic_sta_lta
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
import joblib

# Define paths
data_directory = './data/lunar/training/data/S12_GradeA/'
catalog_path = './data/lunar/training/catalogs/apollo12_catalog_GradeA_final.csv'

# Load the catalog CSV to get filenames and relative event times
catalog_df = pd.read_csv(catalog_path)
# Add .mseed extension to each filename
catalog_files = [f"{filename}.mseed" for filename in catalog_df['filename'].tolist()]
event_times = catalog_df['time_rel(sec)'].tolist()  # This is the target variable

# Prepare lists to hold features and labels
features = []
labels = []

# Filtering parameters
minfreq = 1.0  # Minimum frequency for bandpass filter
maxfreq = 20.0  # Maximum frequency for bandpass filter

# Process each MiniSEED file
for filename, event_time in zip(catalog_files, event_times):
    try:
        print(f"Processing file: {filename}")
        mseed_file = os.path.join(data_directory, filename)
        # Read the MiniSEED file
        st = read(mseed_file)
        tr = st[0]

        # Filter the trace using a bandpass filter
        tr_filtered = tr.copy()
        tr_filtered.filter('bandpass', freqmin=minfreq, freqmax=maxfreq)
        
        # Extract the filtered data and times
        tr_data = tr_filtered.data
        tr_times = tr_filtered.times()
        sampling_rate = tr_filtered.stats.sampling_rate
        trace_duration = tr.stats.endtime - tr.stats.starttime  # Duration in seconds

        # Dynamically set STA/LTA parameters based on trace duration or other characteristics
        sta_len = max(10, trace_duration / 100)  # Ensure a minimum of 1 second for STA
        lta_len = max(600, trace_duration / 10)  # Ensure a minimum of 10 seconds for LTA
        thr_on = 4.0 + (np.std(tr_data) / np.mean(tr_data))  # Example threshold adjustment

        # Compute the STA/LTA characteristic function
        cft = classic_sta_lta(tr_data, int(sta_len * sampling_rate), int(lta_len * sampling_rate))

        # Extract the maximum STA/LTA ratio as a feature
        max_sta_lta = np.max(cft)
        features.append([max_sta_lta])
        labels.append(event_time)

    except Exception as e:
        print(f"Error processing file {filename}: {e}")

# Convert features and labels to numpy arrays for training
features = np.array(features)  # Each feature is the max STA/LTA value
labels = np.array(labels)

# Split the data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

# Define a Random Forest Regressor
model = RandomForestRegressor(random_state=42)

# Set up the hyperparameter grid for tuning
param_grid = {
    'n_estimators': [10, 50, 100],  # Number of trees in the forest
    'max_depth': [None, 10, 20],  # Maximum depth of each tree
    'min_samples_split': [2, 5, 10],  # Minimum number of samples required to split an internal node
    'min_samples_leaf': [1, 2, 4]  # Minimum number of samples required to be at a leaf node
}

# Use GridSearchCV for hyperparameter tuning
grid_search = GridSearchCV(
    estimator=model,
    param_grid=param_grid,
    scoring='neg_mean_absolute_error',
    cv=3,  # 3-fold cross-validation
    n_jobs=-1,  # Use all available cores
    verbose=2  # Show progress
)

# Fit the grid search model
grid_search.fit(X_train, y_train)

# Print the best parameters and the best score
print(f"Best parameters found: {grid_search.best_params_}")
print(f"Best score: {-grid_search.best_score_}")

# Use the best model from the grid search
best_model = grid_search.best_estimator_

# Evaluate the best model on the test set
y_pred = best_model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)

print(f"Mean Absolute Error (MAE) on test set: {mae}")
print(f"Mean Squared Error (MSE) on test set: {mse}")

# Save the best model for later use
joblib.dump(best_model, 'random_forest_sta_lta_best_model_dynamic_params.pkl')
print("Best model training complete and saved.")
