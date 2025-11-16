import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

from hdlib.model.regression import RegressionModel


def main():
    """
    Main function to load data, train the RegHD model, and evaluate its performance.
    """

    # --- 1. HYPERPARAMETERS AND CONFIGURATION ---
    DATASET_PATH = 'environmental_samples.csv'

    TARGET_POLLUTANTS = [
        '[CO2]_ppm',
        '[CO2]d_ppm',
        '[CH4]_ppm',
        '[CH4]d_ppm',
        '[H2O]_ppm'
    ]
    
    # Select the target pollutant to predict.
    TARGET_POLLUTANT = TARGET_POLLUTANTS[0]
    
    # Select the meteorological features to use as input (X).
    METEOROLOGICAL_FEATURES = [
        # Main Gas and Ambient readings
        'GasT_C',
        'GasP_torr',
        'AmbT_C',
        ## Sensor C
        #'C_temp',
        #'C_pre',
        #'C_RH',
        ## Sensor D
        #'D_temp',
        #'D_pre',
        #'D_RH',
        ## Sensor H
        #'H_temp',
        #'H_pre',
        #'H_RH',
        ## Sensor K30
        #'K30_temp',
        #'K30_pre',
        #'K30_RH',
        ## Sensor Vai
        #'Vai_temp',
        #'Vai_pre',
        #'Vai_RH',
        ## Sensor Vai3
        #'Vai3_temp',
        #'Vai3_pre',
        #'Vai3_RH',
        ## Other Temp/RH sensors
        #'RH_12',
        #'T_12',
        #'RH_11',
        #'T_11'
    ]

    # Model Hyperparameters
    HD_DIMENSION = 1000
    K_MODELS = 3
    ITERATIONS = 10
    LEARNING_RATE = 0.01
    TRAIN_SPLIT_RATIO = 0.8 # Use 80% of data for training, 20% for testing

    # --- 2. DATA LOADING AND PREPROCESSING ---
    print(f"--- Loading and Preprocessing Data from '{DATASET_PATH}' ---")
    try:
        df = pd.read_csv(DATASET_PATH)
    except FileNotFoundError:
        print(f"Error: The file '{DATASET_PATH}' was not found.")
        return

    # Select only the columns we need
    columns_to_keep = METEOROLOGICAL_FEATURES + [TARGET_POLLUTANT]
    df_filtered = df[columns_to_keep].copy()

    # --- Handle missing data for time series ---
    # Instead of dropping rows, we will fill missing values using linear interpolation.
    # This preserves the time sequence.
    initial_nas = df_filtered.isna().sum().sum()
    df_filtered.interpolate(method='linear', inplace=True)
    
    # It's possible some NaNs remain at the very start of the file. Drop these if any.
    df_filtered.dropna(inplace=True)

    print(f"Data imputed. Filled {initial_nas} missing values using linear interpolation.")
    print(f"Using {len(df_filtered)} samples for the analysis.")

    if len(df_filtered) == 0:
        print("Error: No valid data remains. Please check the dataset.")
        return

    # To see the correlations with your target specifically:
    correlation_matrix = df_filtered.corr()
    print(correlation_matrix[TARGET_POLLUTANT].sort_values(ascending=False))
    
    # Separate features (X) and target (y)
    X = df_filtered[METEOROLOGICAL_FEATURES].values
    y = df_filtered[TARGET_POLLUTANT].values

    # Split the data before scaling to prevent data leakage
    split_index = int(len(X) * TRAIN_SPLIT_RATIO)
    X_train, X_test = X[:split_index], X[split_index:]
    y_train, y_test = y[:split_index], y[split_index:]

    print(f"Data split sequentially into {len(X_train)} training samples and {len(X_test)} test samples.")

    # Fit the scaler only on the training data
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    # Transform the test data using the scaler fitted on the training data
    X_test_scaled = scaler.transform(X_test)

    print("-" * 30)

    # --- 3. MODEL TRAINING ---
    print("--- Training RegHD Model ---")
    n_features = X_train.shape[1]

    # Initialize the model with our hyperparameters
    model = RegressionModel(
        D=HD_DIMENSION,
        n_features=n_features,
        k_models=K_MODELS,
        learning_rate=LEARNING_RATE,
        iterations=ITERATIONS
    )

    # Enable model quantization
    model.set_quantized_prediction_mode(enable=True)

    # Train the model
    model.fit(X_train, y_train)
    print("-" * 30)

    # --- 4. PREDICTION AND EVALUATION ---
    print("--- Evaluating Model Performance ---")
    y_pred = model.predict(X_test)

    # Calculate performance metrics
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    residuals = y_test - y_pred

    print(f"Mean Squared Error (MSE): {mse:.4f}")
    print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
    print(f"R-squared (R²): {r2:.4f}")
    print("-" * 30)

    # --- 5. VISUALIZATION ---
    print("--- Generating Performance Plots ---")
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, axes = plt.subplots(3, 1, figsize=(12, 20))
    fig.suptitle(f"RegHD Model Performance for '{TARGET_POLLUTANT}'", fontsize=20)

    # Plot 1: Actual vs. Predicted Scatter Plot
    axes[0].scatter(y_test, y_pred, alpha=0.6, edgecolors='k')
    min_val = min(y_test.min(), y_pred.min())
    max_val = max(y_test.max(), y_pred.max())
    axes[0].plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction')
    axes[0].set_xlabel("Actual Values", fontsize=12)
    axes[0].set_ylabel("Predicted Values", fontsize=12)
    axes[0].set_title("Actual vs. Predicted Values", fontsize=16)
    axes[0].legend()
    axes[0].grid(True)

    # Plot 2: Time Series Plot
    axes[1].plot(y_test, label='Actual Values', color='dodgerblue', linewidth=2)
    axes[1].plot(y_pred, label='Predicted Values', color='darkorange', linestyle='--', linewidth=2)
    axes[1].set_xlabel("Time Step (in test set)", fontsize=12)
    axes[1].set_ylabel("Pollutant Concentration", fontsize=12)
    axes[1].set_title("Predictions Over Time", fontsize=16)
    axes[1].legend()
    axes[1].grid(True)
    
    # Plot 3: Residuals Plot
    axes[2].scatter(y_pred, residuals, alpha=0.6, edgecolors='k')
    axes[2].axhline(y=0, color='r', linestyle='--', lw=2)
    axes[2].set_xlabel("Predicted Values", fontsize=12)
    axes[2].set_ylabel("Residuals (Actual - Predicted)", fontsize=12)
    axes[2].set_title("Residuals vs. Predicted Values", fontsize=16)
    axes[2].grid(True)

    plt.tight_layout(rect=[0, 0.03, 1, 0.96])
    
    # Save the figure and show it
    plt.savefig("regression_performance_plots.png")
    print("Plots saved as 'regression_performance_plots.png'")
    plt.show()


if __name__ == "__main__":
    main()
