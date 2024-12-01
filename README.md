# QLR (Quantum Linear Regression)
## Summary
In the qlr.py file, we have implemented the quantum linear regression algorithm which can be found in great detail [here](https://arxiv.org/abs/1402.0660). We broke this algorithm down into the 4 steps listed below:

### Step 1: Load and Normalizing the DataSet
In this phase, we just load a CSV dataset which in our case is a 5-minute intraday report for the TSLA stock. Next, we use the MinMaxScaler() from sklearn to normalize the DataFrame.

### Step 2: Creating Amplitude-Encoded Quantum States
In this phase, we take a normalized row of data and pad it to fit the $2^n$ qubits (where $n = \log_{2}{len(row)}$). Once the data is padded we apply the phases to each qubit.

### Step 3: Simulating the Quantum Circuit
In this phase, we run the quantum simulator, and the amount of shots taken is given by $\frac{1}{\epsilon}$ where $\epsilon$ is the precision amount. For this code, we chose $\epsilon = 1 \times 10^{-3}$.

### Step 4: Implementing the Regression Algorithm
In this phase, we take the normalized matrix of features $X$ and the normalized vector of training results $y$, we then go through each row in $X$ and repeat [Step 3](https://github.com/williamesanders05/QLR/edit/main/README.md#step-3-simulating-the-quantum-circuit) given the quantum circuit created in [Step 2](https://github.com/williamesanders05/QLR/edit/main/README.md#step-2-creating-amplitude-encoded-quantum-states) and the precision amount mentioned in [Step 3](https://github.com/williamesanders05/QLR/edit/main/README.md#step-3-simulating-the-quantum-circuit). This returns coefficients for each feature in the form of probabilities.

## Quantum vs. Classical Linear Regression
We trained both models on the first 100 5-minute intervals of the TSLA stock to see how the open, high, and low values can predict the close value. We then test both models on the next 100 5-minute intervals.
### Accuracy Results:
![Figure 1](https://github.com/williamesanders05/QLR/blob/main/figures/Figure_1.png?raw=true)
### Time Complexity and Timing Results:
Quantum Algorithm Time Complexity:

$O({\kappa}^{3} \times polylog(\frac{d\kappa}{\epsilon}))$

where:

$\kappa =$ Condition number of $X$

$d =$ Number of features

$\epsilon =$ Desired precision

Classical Algorithm Time Complexity:

$O(d^{3} + Nd^{2})$

where:

$N =$ Number of rows in the dataset

#### Here are the Time results for training on 100 rows with 3 features and 1 target:
![Figure 2](https://github.com/williamesanders05/QLR/blob/main/figures/Figure_2.png?raw=true)
