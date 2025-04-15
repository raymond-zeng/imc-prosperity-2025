import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import newton
from scipy.stats import norm
from scipy.optimize import curve_fit

# Load the data
df = pd.read_csv('round-3-island-data-bottle/prices_round_3_day_2.csv', sep=';')

# Function to calculate Black-Scholes call option price
def black_scholes_call(S, K, T, r, sigma):
    d1 = (np.log(S/K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)

# Function to find implied volatility using Newton's method
def implied_volatility(price, S, K, T, r, initial_vol=0.3):
    def objective(sigma):
        return black_scholes_call(S, K, T, r, sigma) - price
    
    try:
        return newton(objective, initial_vol, tol=1e-5, maxiter=100)
    except:
        return np.nan

# Function to fit a parabola
def parabola(x, a, b, c):
    return a * x**2 + b * x + c

# Define parameters
TTE = 6 / 365  # 6 days remaining
r = 0.01  # risk-free rate

# List of strike prices for volcanic rock vouchers
strikes = [9500, 9750, 10000, 10250, 10500]

# Create empty lists to store results
moneyness = []
implied_vols = []
timestamps = []
underlying_prices = []

# Process data for each timestamp
for timestamp in df['timestamp'].unique():
    print(timestamp)
    # Get volcanic rock price
    rock_data = df[(df['timestamp'] == timestamp) & (df['product'] == 'VOLCANIC_ROCK')]
    if not rock_data.empty:
        # Calculate mid price
        bid_price = rock_data['bid_price_1'].iloc[0]
        ask_price = rock_data['ask_price_1'].iloc[0]
        S = (bid_price + ask_price) / 2
        
        # Process each voucher strike
        for K in strikes:
            voucher_symbol = f'VOLCANIC_ROCK_VOUCHER_{K}'
            voucher_data = df[(df['timestamp'] == timestamp) & (df['product'] == voucher_symbol)]
            
            if not voucher_data.empty:
                # Calculate mid price of voucher
                voucher_bid = voucher_data['bid_price_1'].iloc[0]
                voucher_ask = voucher_data['ask_price_1'].iloc[0]
                V = (voucher_bid + voucher_ask) / 2
                
                # Calculate moneyness
                m = np.log(K/S) / np.sqrt(TTE)
                
                # Calculate implied volatility
                iv = implied_volatility(V, S, K, TTE, r)
                
                # Store results
                moneyness.append(m)
                implied_vols.append(iv)
                timestamps.append(timestamp)
                underlying_prices.append(S)

# Create DataFrame with results
results = pd.DataFrame({
    'Timestamp': timestamps,
    'Moneyness': moneyness,
    'ImpliedVol': implied_vols,
    'UnderlyingPrice': underlying_prices
})

# Filter out any invalid volatilities
results = results.dropna()

# Fit a parabolic curve to the data
valid_idx = ~np.isnan(results['ImpliedVol'])
params, _ = curve_fit(parabola, results['Moneyness'][valid_idx], results['ImpliedVol'][valid_idx])

# Generate points for the fitted curve
x_fit = np.linspace(min(results['Moneyness']), max(results['Moneyness']), 100)
y_fit = parabola(x_fit, *params)

# Create the plot
plt.figure(figsize=(12, 8))
plt.scatter(results['Moneyness'], results['ImpliedVol'], alpha=0.5, label='Implied Volatilities')
plt.plot(x_fit, y_fit, 'r-', label=f'Fitted Curve: {params[0]:.4f}x² + {params[1]:.4f}x + {params[2]:.4f}')

# Base IV (at-the-money)
base_iv = parabola(0, *params)
plt.axhline(y=base_iv, color='g', linestyle='--', label=f'Base IV (ATM): {base_iv:.4f}')
plt.axvline(x=0, color='k', linestyle='--', alpha=0.3)

plt.xlabel('Moneyness (m = log(K/S)/sqrt(T))')
plt.ylabel('Implied Volatility')
plt.title('Volatility Smile for Volcanic Rock Vouchers')
plt.legend()
plt.grid(True, alpha=0.3)

plt.show()

# Save the results to a CSV file
results.to_csv('implied_volatility_results.csv', index=False)