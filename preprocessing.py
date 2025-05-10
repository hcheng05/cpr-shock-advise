import os
import numpy as np
import wfdb

def process_record(record_id, data_dir='.', output_dir='processed_data', fs=250, segment_duration=8):
    """
    Process a PhysioNet record into 8-second segments with rhythm labels.

    Parameters:
    - record_id: str, the base name of the record (e.g., '418')
    - data_dir: str, folder where the .dat/.hea/.atr files are located
    - output_dir: str, folder where processed data will be saved
    - fs: int, sampling frequency (default = 250 Hz)
    - segment_duration: int, duration of each segment in seconds

    Returns:
    - segments: np.ndarray of shape (N, segment_samples, 2)
    - labels: np.ndarray of string labels (N,)
    """
    # Derived parameters
    samples_per_segment = fs * segment_duration

    # Load ECG and annotation data
    record_path = os.path.join(data_dir, record_id)
    record = wfdb.rdrecord(record_path)
    annotation = wfdb.rdann(record_path, 'atr')

    # Clean annotations
    aux_notes = [note.replace('\x00', '').lstrip('(').strip().upper() for note in annotation.aux_note]
    sample_indices = annotation.sample

    # Prepare output containers
    signal = record.p_signal  # shape: (total_samples, 2)
    total_segments = signal.shape[0] // samples_per_segment

    segments = []
    labels = []
    last_label = None

    for i in range(total_segments):
        start = i * samples_per_segment
        end = start + samples_per_segment
        segment = signal[start:end, :]

        # Get annotation(s) in this segment
        labels_in_segment = [
            aux_notes[j]
            for j in range(len(sample_indices))
            if start <= sample_indices[j] < end and aux_notes[j] != ''
        ]

        if labels_in_segment:
            label = labels_in_segment[0]  # Or: most common label
            last_label = label
        elif last_label is not None:
            label = last_label
        else:
            label = 'UNKNOWN'

        segments.append(segment)
        labels.append(label)

    # Convert to arrays
    segments = np.array(segments)               # shape: (N, 2000, 2)
    labels = np.array(labels, dtype=object)     # shape: (N,), string labels

    # Make sure output folder exists
    os.makedirs(output_dir, exist_ok=True)

    # Save to .npy files
    np.save(os.path.join(output_dir, f'{record_id}_segments.npy'), segments)
    np.save(os.path.join(output_dir, f'{record_id}_labels.npy'), labels)

    print(f"[✓] Processed '{record_id}': {len(segments)} segments saved.")

    return segments, labels


if __name__ == "__main__":
    record_ids = ['418', '419', '420', '421', '422', '423', '424', '425', '426', '427', '428', '429', '430', '602', '605', '607', '609', '610', '611', '612', '614', '615']
    data_dir = 'data/mit-bih-malignant-ventricular-ectopy-database-1.0.0'
    output_dir = 'processed_data'
    for record_id in record_ids:
        process_record(record_id, data_dir=data_dir, output_dir=output_dir)
    print("[✓] All records processed.")