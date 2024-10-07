import os
import numpy as np
import pandas as pd
from obspy import read
from obspy.signal.trigger import classic_sta_lta
import joblib

# Load the trained model
model = joblib.load('random_forest_sta_lta_best_model_dynamic_params.pkl')

# Define paths
test_data_directory = './data/lunar/test/data/'  #directory of your choide
output_catalog_path = './data/lunar/test/test_event_catalog.csv' #path of your choice

# Prepare a list to hold catalog entries
catalog_entries = []

# Parameters for STA/LTA
minfreq = 0.5  # Minimum frequency for bandpass filter
maxfreq = 1.0  # Maximum frequency for bandpass filter
sta_len_default = 2  # Default STA window length in seconds
lta_len_default = 20  # Default LTA window length in seconds
thr_on = 3  # Threshold for detecting events

# Function to preprocess and check for events
def preprocess_and_detect(file_path):
    try:
        # Read the MiniSEED file
        st = read(file_path)
        tr = st[0]

        # Filter the trace using a bandpass filter
        tr_filtered = tr.copy()
        tr_filtered.filter('bandpass', freqmin=minfreq, freqmax=maxfreq)
        
        # Extract the filtered data and sampling rate
        tr_data = tr_filtered.data
        sampling_rate = tr_filtered.stats.sampling_rate

        # Dynamically set STA/LTA parameters
        trace_duration = tr.stats.endtime - tr.stats.starttime
        sta_len = max(sta_len_default, trace_duration / 100)
        lta_len = max(lta_len_default, trace_duration / 10)

        # Compute the STA/LTA characteristic function
        cft = classic_sta_lta(tr_data, int(sta_len * sampling_rate), int(lta_len * sampling_rate))

        # Detect spikes in the characteristic function
        if np.max(cft) >= thr_on:
            return tr_filtered, cft
        else:
            print(f"No significant event detected in {file_path}.")
            return None, None

    except Exception as e:
        print(f"Error processing file {file_path}: {e}")
        return None, None

# Function to run the model on detected events and add to the catalog
def process_file(file_path, tr_filtered, cft, event_id):
    try:
        # Extract features (e.g., max STA/LTA ratio)
        max_sta_lta = np.max(cft)
        features = np.array([[max_sta_lta]])

        # Make a prediction using the trained model
        predicted_event_time = model.predict(features)

        # Get the starting time of the trace
        start_time = tr_filtered.stats.starttime
        predicted_absolute_time = start_time + predicted_event_time[0]

        # Prepare catalog entry
        catalog_entry = {
            'filename': os.path.splitext(os.path.basename(file_path))[0],
            'time_abs': predicted_absolute_time.isoformat(),
            'time_rel(s)': predicted_event_time[0],
            'evid': event_id
        }
        return catalog_entry

    except Exception as e:
        print(f"Error processing file {file_path} with the model: {e}")
        return None

# Iterate through all files in the test data directory
event_count = 1  # Counter for generating event IDs
for root, _, files in os.walk(test_data_directory):
    for file in files:
        if file.endswith('.mseed'):
            file_path = os.path.join(root, file)
            print(f"Processing file: {file_path}")

            # Preprocess the file and detect events
            tr_filtered, cft = preprocess_and_detect(file_path)

            # If an event is detected, run the model and add to the catalog
            if tr_filtered is not None and cft is not None:
                event_id = f"evid{str(event_count).zfill(5)}"
                catalog_entry = process_file(file_path, tr_filtered, cft, event_id)
                if catalog_entry:
                    catalog_entries.append(catalog_entry)
                    print(f"File {file} processed and added to the catalog as {event_id}.")
                    event_count += 1

# Create a DataFrame from the catalog entries and save to CSV
if catalog_entries:
    catalog_df = pd.DataFrame(catalog_entries)
    catalog_df.to_csv(output_catalog_path, index=False)
    print(f"Catalog saved to {output_catalog_path}")
else:
    print("No events detected in the test data.")

print("Processing complete.")
