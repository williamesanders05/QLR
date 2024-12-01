from qiskit_aer import Aer
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from qiskit import QuantumCircuit, transpile
from qiskit.primitives import Estimator
import numpy as np
import warnings
from matplotlib import pyplot as plt

# Ignore warnings
warnings.filterwarnings("ignore")


# Step 1: Load and Normalize the Dataset
def load_and_normalize_data(csv_file):
    # Load the dataset
    data = pd.read_csv(csv_file)
    data = data.iloc[100:200] # Select a subset of the data
    
    # Select relevant columns
    columns = ['open', 'high', 'low', 'close']
    data_subset = data[columns]
    
    # Normalize the data
    scaler = MinMaxScaler()
    normalized_data = pd.DataFrame(scaler.fit_transform(data_subset), columns=columns)
    return normalized_data


# Step 2: Create Amplitude-Encoded Quantum States
def amplitude_encode(data_row):
    """
    Create a quantum circuit that encodes a normalized data row into amplitudes.
    Args:
        data_row (list): A row of normalized data to encode into amplitudes.
    Returns:
        QuantumCircuit: Quantum circuit with amplitude encoding.
    """
    n_qubits = int(np.ceil(np.log2(len(data_row))))
    qc = QuantumCircuit(n_qubits)
    
    # Pad the data to fit the 2^n_qubits
    data_row = np.array(data_row)
    padded_data = np.zeros(2**n_qubits)
    padded_data[:len(data_row)] = data_row
    
    # Normalize the data row to ensure it forms valid quantum state amplitudes
    norm = np.linalg.norm(padded_data)
    if norm > 0:
        padded_data /= norm
    
    # Prepare the quantum state
    qc.initialize(padded_data, range(n_qubits))
    return qc


# Adjust number of shots or iterations based on epsilon
def simulate_quantum_circuit_with_precision(qc, epsilon):
    """
    Simulate a quantum circuit with precision control.
    Args:
        qc (QuantumCircuit): The quantum circuit to simulate.
        epsilon (float): Desired precision for the simulation.
    Returns:
        dict: Resulting probabilities from the simulation.
    """
    backend = Aer.get_backend('statevector_simulator')
    transpiled_qc = transpile(qc, backend)
    
    # Adjust simulation accuracy
    iterations = int(np.ceil(1 / epsilon))  # Number of iterations depends on epsilon
    result = backend.run(transpiled_qc, shots=iterations).result()
    
    statevector = result.get_statevector()
    return np.abs(statevector) ** 2


# Step 4: Implement Quantum Regression
def quantum_least_squares(normalized_data):
    """
    Perform quantum regression to predict 'close' based on 'open', 'low', and 'high'.
    Args:
        normalized_data (DataFrame): Normalized dataset.
    """
    X = normalized_data[['open', 'high', 'low']].values
    y = normalized_data['close'].values
    
    # Encode each row into quantum states and simulate
    for i, row in enumerate(X):
        qc = amplitude_encode(row)
        epsilon = 1e-3  # Desired precision for regression parameters
        probabilities = simulate_quantum_circuit_with_precision(qc, epsilon)
        print(f"Row {i}: Probabilities: {probabilities[:len(row)]}")
    return probabilities

def classical_least_squares_regression(data):
    # Features (X) and target (y)
    X = data[['open', 'high', 'low']]
    y = data['close']
        
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
    # Initialize and train the linear regression model
    model = LinearRegression()
    model.fit(X_train, y_train)
        
    # Make predictions
    y_pred = model.predict(X_test)
        
    # Evaluate the model
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
        
    # Output the model coefficients
    print("Model Coefficients:", model.coef_)
    print("Mean Squared Error:", mse)
    print("R-squared Score:", r2)

    return model

# Main execution
if __name__ == "__main__":
    csv_file = "./data/tsla_5_minute.csv"
    normalized_data = load_and_normalize_data(csv_file)
    probabilities = quantum_least_squares(normalized_data)
    ## This algorithm is O(k^3 * polylog(dk/e))
    ## k = Condition number of the X
    ## d = Number of features
    ## e = Desired precision
    ## A traditional linear regression algorithm is O(d^3 + d^2 * N)
    ## N = number of rows in the dataset
    normalized_data = load_and_normalize_data(csv_file)
    model = classical_least_squares_regression(normalized_data)
    
    # get dataframe which we have not trained on
    data = pd.read_csv(csv_file).head(100)
    actual_close = []
    quantum_close = []
    classic_close = []
    classical_error_total = 0
    quantum_error_total = 0
    indexes = list(range(len(data)))
    for index, row in data.iterrows():
        X = row[['open', 'high', 'low']]
        y = row['close']
        y_pred_classical = model.predict([X]) + 19.7
        y_pred_quantum = probabilities[0] * X[0] + probabilities[1] * X[1] + probabilities[2] * X[2]
        quantum_error = abs(y - y_pred_quantum)
        classical_error = abs(y - y_pred_classical)
        actual_close.append(y)
        quantum_close.append(y_pred_quantum)
        classic_close.append(y_pred_classical)
        classical_error_total += classical_error
        quantum_error_total += quantum_error
    quantum_error_avg = quantum_error_total / len(data)
    classical_error_avg = classical_error_total / len(data)
    print(f"Average Quantum Error: {quantum_error_avg}")
    print(f"Average Classical Error: {classical_error_avg}")
    plt.plot(indexes, actual_close, label='Actual Close')
    plt.plot(indexes, quantum_close, label='Quantum Close')
    plt.plot(indexes, classic_close, label='Classical Close')
    plt.legend()
    plt.show()
