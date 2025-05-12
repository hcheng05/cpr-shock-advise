import numpy as np
import os

# List of record IDs
record_ids = ['418', '419', '420', '421', '422', '423', '424', '425', '426', '427', '428', '429', '430', '602', '605', '607', '609', '610', '611', '612', '614', '615']  # add your IDs
data_dir = 'processed_data'

# Initialize lists
all_segments = []
all_labels = []

for rec_id in record_ids:
    seg_path = os.path.join(data_dir, f'{rec_id}_segments.npy')
    lab_path = os.path.join(data_dir, f'{rec_id}_labels.npy')
    
    segments = np.load(seg_path).astype(np.float32)
    labels = np.load(lab_path).astype(np.int32)
    
    all_segments.append(segments)
    all_labels.append(labels)

# Concatenate all arrays
X = np.concatenate(all_segments, axis=0)  # shape: (total_segments, 2000, 2)
y = np.concatenate(all_labels, axis=0)    # shape: (total_segments,)

# Save the combined data
os.makedirs(data_dir, exist_ok=True)
np.save('all_segments.npy', X)
np.save('all_labels.npy', y)
