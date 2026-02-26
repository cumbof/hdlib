"""MNIST Classification.
Data retrieval and preprocessing borrowed from https://www.tensorflow.org/quantum/tutorials/mnist"""

import collections
import copy
import os
import time # Import the time module
import json # Import the json module for checkpoints

import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix, roc_curve
from sklearn.model_selection import KFold

from hdlib.model import ClassificationModel, QuantumClassificationModel

from qiskit_algorithms.optimizers import COBYLA
from qiskit.circuit.library import ZFeatureMap, ZZFeatureMap, RealAmplitudes
from qiskit_machine_learning.algorithms.classifiers import VQC, QSVC
from qiskit_machine_learning.state_fidelities import ComputeUncompute
from qiskit_machine_learning.kernels import FidelityQuantumKernel
from qiskit.primitives import StatevectorSampler


# Configuration for Hardware (Optional)
CHANNEL = "IBM-CHANNEL"
INSTANCE = "IBM-INSTANCE"
BACKEND = "IBM-BACKEND"
API_KEY = "YOUR-API-KEY"

def print_cv_summary(model_name, reports, matrices, times, n_splits):
    """Prints a summary of cross-validation results."""
    print(f"\n--- {model_name} CV Summary ({n_splits}-Folds) ---")

    if not reports:
        print("No reports to summarize.")

        return

    # Average and std dev of time
    try:
        avg_time = np.mean(times)
        std_time = np.std(times)
        print(f"Average Total Time (fit+predict): {avg_time:.2f} ± {std_time:.2f} sec")

    except Exception as e:
        print(f"Could not calculate average time: {e}")

    # Sum confusion matrices
    try:
        total_cm = np.sum(matrices, axis=0)
        print(f"Total Confusion Matrix (sum over {n_splits} folds):\n{total_cm}")

    except Exception as e:
        print(f"Could not sum confusion matrices: {e}")

    # Average metrics from reports
    try:
        # avg_accuracy = np.mean([r['accuracy'] for r in reports])

        # Calculate average and std dev for F1 scores
        macro_f1_scores = [r['macro avg']['f1-score'] for r in reports]
        avg_macro_f1 = np.mean(macro_f1_scores)
        std_macro_f1 = np.std(macro_f1_scores)

        weighted_f1_scores = [r['weighted avg']['f1-score'] for r in reports]
        avg_weighted_f1 = np.mean(weighted_f1_scores)
        std_weighted_f1 = np.std(weighted_f1_scores)

        # Handle cases where a class might not be present in a fold's prediction
        precision_6_scores = [r['Digit 6']['precision'] for r in reports if 'Digit 6' in r]
        avg_precision_6 = np.mean(precision_6_scores) if precision_6_scores else 0
        std_precision_6 = np.std(precision_6_scores) if precision_6_scores else 0

        recall_6_scores = [r['Digit 6']['recall'] for r in reports if 'Digit 6' in r]
        avg_recall_6 = np.mean(recall_6_scores) if recall_6_scores else 0
        std_recall_6 = np.std(recall_6_scores) if recall_6_scores else 0

        f1_6_scores = [r['Digit 6']['f1-score'] for r in reports if 'Digit 6' in r]
        avg_f1_6 = np.mean(f1_6_scores) if f1_6_scores else 0
        std_f1_6 = np.std(f1_6_scores) if f1_6_scores else 0

        precision_3_scores = [r['Digit 3']['precision'] for r in reports if 'Digit 3' in r]
        avg_precision_3 = np.mean(precision_3_scores) if precision_3_scores else 0
        std_precision_3 = np.std(precision_3_scores) if precision_3_scores else 0

        recall_3_scores = [r['Digit 3']['recall'] for r in reports if 'Digit 3' in r]
        avg_recall_3 = np.mean(recall_3_scores) if recall_3_scores else 0
        std_recall_3 = np.std(recall_3_scores) if recall_3_scores else 0

        f1_3_scores = [r['Digit 3']['f1-score'] for r in reports if 'Digit 3' in r]
        avg_f1_3 = np.mean(f1_3_scores) if f1_3_scores else 0
        std_f1_3 = np.std(f1_3_scores) if f1_3_scores else 0

        # print(f"Average Accuracy: {avg_accuracy:.4f}")
        print(f"Average Macro F1-Score:     {avg_macro_f1:.4f} ± {std_macro_f1:.4f}")
        print(f"Average Weighted F1-Score: {avg_weighted_f1:.4f} ± {std_weighted_f1:.4f}")

        print("\nAverage Metrics for 'Digit 6':")
        print(f"  Precision: {avg_precision_6:.4f} ± {std_precision_6:.4f}")
        print(f"  Recall:    {avg_recall_6:.4f} ± {std_recall_6:.4f}")
        print(f"  F1-Score:  {avg_f1_6:.4f} ± {std_f1_6:.4f}")

        print("\nAverage Metrics for 'Digit 3':")
        print(f"  Precision: {avg_precision_3:.4f} ± {std_precision_3:.4f}")
        print(f"  Recall:    {avg_recall_3:.4f} ± {std_recall_3:.4f}")
        print(f"  F1-Score:  {avg_f1_3:.4f} ± {std_f1_3:.4f}")

    except Exception as e:
        print(f"Could not calculate average metrics: {e}")
        print("Note: This can happen if one class was not predicted in a fold.")


if __name__ == "__main__":
    print("--- 1. Preparing MNIST Dataset ---")

    # Retrieve the MNIST dataset
    (X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()

    # Rescale the images from [0,255] to the [0.0,1.0] range
    X_train, X_test = X_train[..., np.newaxis] / 255.0, X_test[..., np.newaxis] / 255.0

    # Filter the dataset to keep just the 3s and 6s, remove the other classes
    def filter_3_6(X, y):
        keep = (y == 3) | (y == 6)
        X, y = X[keep], y[keep]
        # Convert labels to integers: 1 for digit '3', 0 for digit '6'
        y = (y == 3).astype(int)
        return X, y

    X_train, y_train = filter_3_6(X_train, y_train)
    X_test, y_test = filter_3_6(X_test, y_test)

    print(f"Filtered training set size: {len(X_train)}")
    print(f"Filtered test set size: {len(X_test)}")

    # Downscale the images
    # An image size of 28x28 is much too large for current quantum computers. Resize the image down to 4x4
    X_train_small = tf.image.resize(X_train, (4, 4)).numpy()
    X_test_small = tf.image.resize(X_test, (4, 4)).numpy()

    # Remove contradictory examples
    # Filter the dataset to remove images that are labeled as belonging to both classes
    def remove_contradicting(Xs, ys):
        mapping = collections.defaultdict(set)
        orig_X = dict()

        # Determine the set of labels for each unique image
        for X, y in zip (Xs, ys):
            flat_X = tuple(X.flatten())
            orig_X[flat_X] = X
            mapping[flat_X].add(y)

        new_X, new_y = list(), list()

        for flatten_X in mapping:
            X = orig_X[flatten_X]
            labels = mapping[flatten_X]

            if len(labels) == 1:
                new_X.append(X)
                new_y.append(next(iter(labels)))

        return np.array(new_X), np.array(new_y)

    # This is not a standard machine-learning procedure, but it is included in the interest of following the paper:
    # Farhi, E et al. "Classification with quantum neural networks on near term processors". arXiv (2018). https://doi.org/10.48550/arXiv.1802.06002
    X_train_nocon, y_train_nocon = remove_contradicting(X_train_small, y_train)

    print(f"Training set size after removing contradictions: {len(X_train_nocon)}")

    print("\n--- 2. Binarizing Images and Formatting Data ---")

    # Convert grayscale images to black and white
    # Apply a threshold to binarize the pixel values
    threshold = 0.5
    X_train_bin = (X_train_nocon >= threshold).astype(float)
    X_test_bin = (X_test_small >= threshold).astype(float)

    print(f"Pixel values binarized using a {threshold} threshold")

    # Our QuantumClassificationModel expects a flat list of features for each sample
    # Reshape the (4, 4, 1) images into (16,) vectors
    X_train_flat = X_train_bin.reshape(X_train_bin.shape[0], -1)
    X_test_flat = X_test_bin.reshape(X_test_bin.shape[0], -1)

    # The model's fit/predict methods expect lists, not numpy arrays
    X_train_list = X_train_flat.tolist()
    y_train_list = y_train_nocon.tolist()
    X_test_list = X_test_flat.tolist()

    print("Data formatting complete. Sample features are now 16-element vectors of 0s and 1s")

    print("\n--- 3. Subsampling Data and Preparing for Cross-Validation ---")

    # Define the number of samples to keep for training and testing
    n_train = 100
    n_test = 50

    # Use a fixed seed for reproducibility shuffling
    np.random.seed(42)

    # Shuffle the training data to ensure randomness
    shuffle_indices_train = np.random.permutation(len(X_train_list))

    # Apply shuffle to both data and labels
    X_train_shuffled = [X_train_list[i] for i in shuffle_indices_train]
    y_train_shuffled = [y_train_list[i] for i in shuffle_indices_train]

    # Take a small slice of the data for the final training set
    X_train_sub = X_train_shuffled[:n_train]
    y_train_sub = y_train_shuffled[:n_train]

    # Shuffle and slice the test data as well
    shuffle_indices_test = np.random.permutation(len(X_test_list))
    X_test_shuffled = [X_test_list[i] for i in shuffle_indices_test]
    y_test_shuffled = [y_test[i] for i in shuffle_indices_test]

    # Take a small slice of the data for the final test set
    X_test_sub = X_test_shuffled[:n_test]
    y_test_sub = [int(l) for l in y_test_shuffled[:n_test]] # Ensure labels are int

    # Combine subsampled train and test sets for cross-validation
    X_final = X_train_sub + X_test_sub
    y_final = y_train_sub + y_test_sub

    # Convert to numpy array for KFold splitting, though models expect lists
    X_final_np = np.array(X_final)
    y_final_np = np.array(y_final)

    print(f"Combined dataset for cross-validation: {len(X_final)} samples")

    # --- 4. Running 5-Fold Cross-Validation ---
    print("\n--- 4. Initializing 5-Fold Cross-Validation ---")

    N_SPLITS = 5
    kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=42)

    # Lists to store results from each fold
    classical_10k_reports = list()
    classical_10k_matrices = list()
    classical_128_reports = list()
    classical_128_matrices = list()
    quantum_128_reports = list()
    quantum_128_matrices = list()
    vqc_reports = list()
    vqc_matrices = list()
    qsvc_reports = list()
    qsvc_matrices = list()

    # Lists to store timing for each fold
    classical_10k_times = list()
    classical_128_times = list()
    quantum_128_times = list()
    vqc_times = list()
    qsvc_times = list()

    # New lists to store (true_label, score) tuples for ROC curve data
    classical_10k_roc_data = list()
    classical_128_roc_data = list()
    quantum_128_roc_data = list()
    vqc_roc_data = list()
    qsvc_roc_data = list()

    fold_num = 1

    for train_index, test_index in kf.split(X_final_np):        
        checkpoint_file = f"fold_{fold_num}_results.json"

        # --- Checkpoint Loading ---
        # Check if results for this fold already exist
        if os.path.exists(checkpoint_file):
            print(f"\n--- LOADING FOLD {fold_num}/{N_SPLITS} FROM CHECKPOINT ---")

            with open(checkpoint_file, 'r') as f:
                fold_data = json.load(f)

            # Load and append C10K data
            classical_10k_reports.append(fold_data['c10k_report'])
            classical_10k_matrices.append(np.array(fold_data['c10k_matrix']))
            classical_10k_times.append(fold_data['c10k_time'])
            classical_10k_roc_data.extend(fold_data['c10k_roc_data'])

            # Load and append C128 data
            classical_128_reports.append(fold_data['c128_report'])
            classical_128_matrices.append(np.array(fold_data['c128_matrix']))
            classical_128_times.append(fold_data['c128_time'])
            classical_128_roc_data.extend(fold_data['c128_roc_data'])

            # Load and append Q128 data
            quantum_128_reports.append(fold_data['q128_report'])
            quantum_128_matrices.append(np.array(fold_data['q128_matrix']))
            quantum_128_times.append(fold_data['q128_time'])
            quantum_128_roc_data.extend(fold_data['q128_roc_data'])

            # Load and append VQC data
            vqc_reports.append(fold_data['vqc_report'])
            vqc_matrices.append(np.array(fold_data['vqc_matrix']))
            vqc_times.append(fold_data['vqc_time'])
            vqc_roc_data.extend(fold_data['vqc_roc_data'])

            # Load and append QSVC data
            qsvc_reports.append(fold_data['qsvc_report'])
            qsvc_matrices.append(np.array(fold_data['qsvc_matrix']))
            qsvc_times.append(fold_data['qsvc_time'])
            qsvc_roc_data.extend(fold_data['qsvc_roc_data'])

            fold_num += 1

            continue # Skip to the next fold

        # --- If checkpoint not found, run the fold ---
        print(f"\n--- RUNNING FOLD {fold_num}/{N_SPLITS} ---")

        # Create list-based data for this fold
        # The models expect lists, not numpy arrays
        X_train_fold = [X_final[i] for i in train_index]
        y_train_fold = [y_final[i] for i in train_index]
        X_test_fold = [X_final[i] for i in test_index]
        y_test_fold = [y_final[i] for i in test_index]

        # Create numpy versions for Qiskit models
        X_train_fold_np = X_final_np[train_index]
        y_train_fold_np = y_final_np[train_index]
        X_test_fold_np = X_final_np[test_index]
        # y_test_fold_np is y_test_fold (already a list of ints)

        # Prepare data for the classical model's specific fit/predict format
        X_all_fold = X_train_fold + X_test_fold
        y_all_fold = y_train_fold + y_test_fold

        # Get the indices for the test set relative to X_all_fold
        test_idx_fold = list(range(len(X_train_fold), len(X_all_fold)))

        # --- Classical Model (Dimensionality=10000) ---
        print("\nTraining Classical Model (D=10000)...")
        start_time_c10k = time.perf_counter()
        model_c10k = ClassificationModel(size=10000, levels=2)
        model_c10k.fit(X_all_fold, y_all_fold)

        print("Evaluating Classical Model (D=10000)...")
        # Capture the 'similarities' output (index 2)
        _, y_pred_c10k, similarities_c10k, _, _, _ = model_c10k.predict(test_idx_fold, retrain=0)
        end_time_c10k = time.perf_counter()

        # Store this fold's results in variables
        fold_time_c10k = end_time_c10k - start_time_c10k
        fold_report_c10k = classification_report(y_test_fold, y_pred_c10k, target_names=["Digit 6", "Digit 3"], output_dict=True, zero_division=0)
        fold_matrix_c10k = confusion_matrix(y_test_fold, y_pred_c10k, labels=[0, 1]) # Ensure consistent label order

        # Store ROC data points (True Label, Score for Class 1)
        fold_roc_data_c10k = list()
        for true_label, sim_pair in zip(y_test_fold, similarities_c10k):
            # Invert distance (0 to 1) to create a score (0 to 1), where 1 is a perfect match
            # Cast to standard float for JSON serialization
            score_for_class_1 = 1.0 - sim_pair[1] # Using 1.0 - distance_to_positive_class
            fold_roc_data_c10k.append((true_label, float(score_for_class_1)))

        # --- Classical Model (Dimensionality=128) ---
        print("\nTraining Classical Model (D=128)...")
        start_time_c128 = time.perf_counter()
        model_c128 = ClassificationModel(size=128, levels=2)
        model_c128.fit(X_all_fold, y_all_fold) # Uses same X_all_fold, y_all_fold, test_idx_fold

        print("Evaluating Classical Model (D=128)...")
        # Capture the 'similarities' output (index 2)
        _, y_pred_c128, similarities_c128, _, _, _ = model_c128.predict(test_idx_fold, retrain=0)
        end_time_c128 = time.perf_counter()

        # Store this fold's results in variables
        fold_time_c128 = end_time_c128 - start_time_c128
        fold_report_c128 = classification_report(y_test_fold, y_pred_c128, target_names=["Digit 6", "Digit 3"], output_dict=True, zero_division=0)
        fold_matrix_c128 = confusion_matrix(y_test_fold, y_pred_c128, labels=[0, 1]) # Ensure consistent label order

        # Store ROC data points (True Label, Score for Class 1)
        fold_roc_data_c128 = list()
        for true_label, sim_pair in zip(y_test_fold, similarities_c128):
            # Invert distance (0 to 1) to create a score (0 to 1), where 1 is a perfect match
            # Cast to standard float for JSON serialization
            score_for_class_1 = 1.0 - sim_pair[1] # Using 1.0 - distance_to_positive_class
            fold_roc_data_c128.append((true_label, float(score_for_class_1)))

        # --- Quantum Model (Dimensionality=128) ---
        print("\nTraining Quantum Model (D=128)...")
        start_time_q128 = time.perf_counter()

        # Noise-free simulation
        model_q128 = QuantumClassificationModel(size=128, levels=2, shots=10000)

        # Simulation with noise model
        #model_q128 = QuantumClassificationModel(size=128, levels=2, shots=10000, api_key=API_KEY, noise_model_from=BACKEND)

        # Quantum hardware
        #model_q128 = QuantumClassificationModel(size=128, levels=2, shots=10000, channel=CHANNEL, instance=INSTANCE, backend=BACKEND, api_key=API_KEY)

        # One-shot learning
        model_q128.fit(X_train_fold, y_train_fold) # Quantum model uses standard fit/predict

        print("Evaluating Quantum Model (D=128)...")
        y_pred_q128, scores_q128 = model_q128.predict(X_test_fold)
        end_time_q128 = time.perf_counter()

        # Store this fold's results in variables
        fold_time_q128 = end_time_q128 - start_time_q128
        fold_report_q128 = classification_report(y_test_fold, y_pred_q128, target_names=["Digit 6", "Digit 3"], output_dict=True, zero_division=0)
        fold_matrix_q128 = confusion_matrix(y_test_fold, y_pred_q128, labels=[0, 1]) # Ensure consistent label order

        # Store ROC data points (True Label, Score for Class 1)
        fold_roc_data_q128 = list()
        for true_label, score_pair in zip(y_test_fold, scores_q128):
            score_for_class_1 = score_pair[1] # Using score for positive class (Digit 3)
            # Cast to standard float for JSON serialization
            fold_roc_data_q128.append((true_label, float(score_for_class_1)))

        n_features = X_train_fold_np.shape[1] # Should be 16

        # --- QSVC Model (QSVM) ---
        print("\nTraining QSVC Model (QSVM)...")
        start_time_qsvc = time.perf_counter()

        # Setup QSVC feature map
        feature_map_qsvc = ZZFeatureMap(feature_dimension=n_features, reps=1, entanglement='linear')

        sampler = StatevectorSampler()

        # Create the fidelity object
        fidelity = ComputeUncompute(sampler=sampler)

        # Create the QuantumKernel using the fidelity and feature map
        qsvc_kernel = FidelityQuantumKernel(fidelity=fidelity, feature_map=feature_map_qsvc)

        qsvc = QSVC(quantum_kernel=qsvc_kernel)

        qsvc.fit(X_train_fold_np, y_train_fold_np)

        print("Evaluating QSVC Model...")
        y_pred_qsvc = qsvc.predict(X_test_fold_np)
        
        # Get decision function scores for ROC (1D array)
        scores_qsvc = qsvc.decision_function(X_test_fold_np)

        end_time_qsvc = time.perf_counter()

        # Store this fold's results
        fold_time_qsvc = end_time_qsvc - start_time_qsvc
        fold_report_qsvc = classification_report(y_test_fold, y_pred_qsvc, target_names=["Digit 6", "Digit 3"], output_dict=True, zero_division=0)
        fold_matrix_qsvc = confusion_matrix(y_test_fold, y_pred_qsvc, labels=[0, 1])

        scores_qsvc_list = scores_qsvc.tolist()
        fold_roc_data_qsvc = []
        for true_label, score_for_class_1 in zip(y_test_fold, scores_qsvc_list):
            fold_roc_data_qsvc.append((true_label, float(score_for_class_1)))

        # --- VQC Model (QNN) ---
        print("\nTraining VQC Model (QNN)...")
        start_time_vqc = time.perf_counter()
        
        # Setup VQC components
        feature_map_vqc = ZFeatureMap(feature_dimension=n_features, reps=1)
        ansatz_vqc = RealAmplitudes(num_qubits=n_features, reps=3)
        optimizer_vqc = COBYLA(maxiter=100)

        vqc = VQC(
            feature_map=feature_map_vqc,
            ansatz=ansatz_vqc,
            optimizer=optimizer_vqc
        )

        vqc.fit(X_train_fold_np, y_train_fold_np)
        
        print("Evaluating VQC Model...")
        y_pred_vqc = vqc.predict(X_test_fold_np)
        
        # Get probabilities (scores) for ROC. VQC's network forward pass returns (batch_size, num_classes)
        scores_vqc_raw = vqc.neural_network.forward(X_test_fold_np, vqc.weights)
        
        end_time_vqc = time.perf_counter()

        # Store this fold's results
        fold_time_vqc = end_time_vqc - start_time_vqc
        fold_report_vqc = classification_report(y_test_fold, y_pred_vqc, target_names=["Digit 6", "Digit 3"], output_dict=True, zero_division=0)
        fold_matrix_vqc = confusion_matrix(y_test_fold, y_pred_vqc, labels=[0, 1])

        # Get the score for class 1
        scores_vqc_class1 = [prob[1] for prob in scores_vqc_raw.tolist()]
        fold_roc_data_vqc = []
        for true_label, score_for_class_1 in zip(y_test_fold, scores_vqc_class1):
            fold_roc_data_vqc.append((true_label, float(score_for_class_1)))

        # --- Checkpoint Saving ---
        # Collate all results for this fold
        fold_data = {
            'c10k_report': fold_report_c10k,
            'c10k_matrix': fold_matrix_c10k.tolist(),
            'c10k_time': fold_time_c10k,
            'c10k_roc_data': fold_roc_data_c10k,

            'c128_report': fold_report_c128,
            'c128_matrix': fold_matrix_c128.tolist(),
            'c128_time': fold_time_c128,
            'c128_roc_data': fold_roc_data_c128,

            'q128_report': fold_report_q128,
            'q128_matrix': fold_matrix_q128.tolist(),
            'q128_time': fold_time_q128,
            'q128_roc_data': fold_roc_data_q128,

            'vqc_report': fold_report_vqc,
            'vqc_matrix': fold_matrix_vqc.tolist(),
            'vqc_time': fold_time_vqc,
            'vqc_roc_data': fold_roc_data_vqc,

            'qsvc_report': fold_report_qsvc,
            'qsvc_matrix': fold_matrix_qsvc.tolist(),
            'qsvc_time': fold_time_qsvc,
            'qsvc_roc_data': fold_roc_data_qsvc
        }

        # Save this fold's data to its checkpoint file
        with open(checkpoint_file, 'w') as f:
            json.dump(fold_data, f, indent=4)

        print(f"--- SAVED FOLD {fold_num} RESULTS TO {checkpoint_file} ---")

        # --- Append results to main lists for final summary ---
        classical_10k_reports.append(fold_report_c10k)
        classical_10k_matrices.append(fold_matrix_c10k)
        classical_10k_times.append(fold_time_c10k)
        classical_10k_roc_data.extend(fold_roc_data_c10k)

        classical_128_reports.append(fold_report_c128)
        classical_128_matrices.append(fold_matrix_c128)
        classical_128_times.append(fold_time_c128)
        classical_128_roc_data.extend(fold_roc_data_c128)

        quantum_128_reports.append(fold_report_q128)
        quantum_128_matrices.append(fold_matrix_q128)
        quantum_128_times.append(fold_time_q128)
        quantum_128_roc_data.extend(fold_roc_data_q128)

        vqc_reports.append(fold_report_vqc)
        vqc_matrices.append(fold_matrix_vqc)
        vqc_times.append(fold_time_vqc)
        vqc_roc_data.extend(fold_roc_data_vqc)

        qsvc_reports.append(fold_report_qsvc)
        qsvc_matrices.append(fold_matrix_qsvc)
        qsvc_times.append(fold_time_qsvc)
        qsvc_roc_data.extend(fold_roc_data_qsvc)

        fold_num += 1

    # --- 5. Cross-Validation Results Summary ---
    print("\n--- 5. Cross-Validation Results Summary ---")

    print_cv_summary("Classical Model (D=10000)", classical_10k_reports, classical_10k_matrices, classical_10k_times, N_SPLITS)
    print_cv_summary("Classical Model (D=128)", classical_128_reports, classical_128_matrices, classical_128_times, N_SPLITS)
    print_cv_summary("Quantum Model (D=128)", quantum_128_reports, quantum_128_matrices, quantum_128_times, N_SPLITS)
    print_cv_summary("VQC Model (QNN)", vqc_reports, vqc_matrices, vqc_times, N_SPLITS)
    print_cv_summary("QSVC Model (QSVM)", qsvc_reports, qsvc_matrices, qsvc_times, N_SPLITS)

    # --- 6. ROC Curve Data Points ---
    print("\n--- 6. ROC Curve Data Points (True Label, Score) ---")

    print(f"\nClassical Model (D=10000) ROC Data ({len(classical_10k_roc_data)} points):")
    print("[(True Label, Score for Class 1), ...]")
    print(classical_10k_roc_data)

    print(f"\nClassical Model (D=128) ROC Data ({len(classical_128_roc_data)} points):")
    print("[(True Label, Score for Class 1), ...]")
    print(classical_128_roc_data)

    print(f"\nQuantum Model (D=128) ROC Data ({len(quantum_128_roc_data)} points):")
    print("[(True Label, Score for Class 1), ...]")
    print(quantum_128_roc_data)

    print(f"\nVQC Model (QNN) ROC Data ({len(vqc_roc_data)} points):")
    print("[(True Label, Score for Class 1), ...]")
    print(vqc_roc_data)

    print(f"\nQSVC Model (QSVM) ROC Data ({len(qsvc_roc_data)} points):")
    print("[(True Label, Score for Class 1), ...]")
    print(qsvc_roc_data)

    # --- 7. Calculate Exact ROC Plotting Points ---
    print("\n--- 7. Exact (FPR, TPR) Points for Plotting ---")
    print("Calculated using sklearn.metrics.roc_curve")

    try:
        # --- Classical Model (D=10000) ---
        if classical_10k_roc_data:
            # Unzip the (true_label, score) tuples into separate lists
            y_true_c10k = [item[0] for item in classical_10k_roc_data]
            y_scores_c10k = [item[1] for item in classical_10k_roc_data]

            # Calculate the ROC curve points
            fpr_c10k, tpr_c10k, _ = roc_curve(y_true_c10k, y_scores_c10k)

            # Create a list of (x, y) coordinates
            roc_points_c10k = list(zip(fpr_c10k, tpr_c10k))

            print(f"\nClassical Model (D=10000) ROC Plot Points ({len(roc_points_c10k)} points):")
            print("[(FPR, TPR), ...]")
            print(roc_points_c10k)

        # --- Classical Model (D=128) ---
        if classical_128_roc_data:
            y_true_c128 = [item[0] for item in classical_128_roc_data]
            y_scores_c128 = [item[1] for item in classical_128_roc_data]
            fpr_c128, tpr_c128, _ = roc_curve(y_true_c128, y_scores_c128)
            roc_points_c128 = list(zip(fpr_c128, tpr_c128))

            print(f"\nClassical Model (D=128) ROC Plot Points ({len(roc_points_c128)} points):")
            print("[(FPR, TPR), ...]")
            print(roc_points_c128)

        # --- Quantum Model (D=128) ---
        if quantum_128_roc_data:
            y_true_q128 = [item[0] for item in quantum_128_roc_data]
            y_scores_q128 = [item[1] for item in quantum_128_roc_data]
            fpr_q128, tpr_q128, _ = roc_curve(y_true_q128, y_scores_q128)
            roc_points_q128 = list(zip(fpr_q128, tpr_q128))

            print(f"\nQuantum Model (D=128) ROC Plot Points ({len(roc_points_q128)} points):")
            print("[(FPR, TPR), ...]")
            print(roc_points_q128)

        # --- VQC Model (QNN) ---
        if vqc_roc_data:
            y_true_vqc = [item[0] for item in vqc_roc_data]
            y_scores_vqc = [item[1] for item in vqc_roc_data]
            fpr_vqc, tpr_vqc, _ = roc_curve(y_true_vqc, y_scores_vqc)
            roc_points_vqc = list(zip(fpr_vqc, tpr_vqc))
            
            print(f"\nVQC Model (QNN) ROC Plot Points ({len(roc_points_vqc)} points):")
            print("[(FPR, TPR), ...]")
            print(roc_points_vqc)

        # --- QSVC Model (QSVM) ---
        if qsvc_roc_data:
            y_true_qsvc = [item[0] for item in qsvc_roc_data]
            y_scores_qsvc = [item[1] for item in qsvc_roc_data]
            fpr_qsvc, tpr_qsvc, _ = roc_curve(y_true_qsvc, y_scores_qsvc)
            roc_points_qsvc = list(zip(fpr_qsvc, tpr_qsvc))
            
            print(f"\nQSVC Model (QSVM) ROC Plot Points ({len(roc_points_qsvc)} points):")
            print("[(FPR, TPR), ...]")
            print(roc_points_qsvc)

    except Exception as e:
        print(f"\nCould not calculate ROC curve points: {e}")
        print("This can happen if one of the classes is not present in the test data.")
