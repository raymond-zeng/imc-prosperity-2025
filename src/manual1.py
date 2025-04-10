
import numpy as np
# Define the trade matrix
trade_matrix = np.array([
    [1, 1.45, 0.52, 0.72],
    [0.7, 1, 0.31, 0.48],
    [1.95, 3.1, 1, 1.49],
    [1.34, 1.98, 0.64, 1]
])

# Number of items
n = trade_matrix.shape[0]

# Number of trades
k = 5

# Initialize dynamic programming table
# dp[i][j] = max value of item j obtainable in i trades
dp = np.zeros((k + 1, n))
dp[0][3] = 500000  # Start with 1 seashell (index 3)

# Fill DP table
for step in range(1, k + 1):
    for to_item in range(n):
        for from_item in range(n):
            if dp[step - 1][from_item] * trade_matrix[from_item][to_item] > dp[step][to_item]:
                dp[step][to_item] = dp[step - 1][from_item] * trade_matrix[from_item][to_item]
                print(f"Step {step}: Trade from {from_item} to {to_item}, Value: {dp[step][to_item]:.2f}")



                                    

max_seashells = dp[k][3]  # Max value of seashells after k trades

print(f"Max seashells after {k} trades: {max_seashells:.2f}")

print("dp table:")
for i in range(k + 1):
    print(f"Step {i}: {dp[i]}")