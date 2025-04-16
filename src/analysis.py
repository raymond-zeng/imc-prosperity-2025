import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import newton
from scipy.stats import norm
from scipy.optimize import curve_fit

# Load the data
df = pd.read_csv('round-3-island-data-bottle/prices_round_3_day_0.csv', sep=';')
df = pd.read_csv('round-3-island-data-bottle/prices_round_3_day_1.csv', sep=';')
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
TTE = 8 / 365  
r = 0.00  # risk-free rate

# List of strike prices for volcanic rock vouchers
strikes = [9500, 9750, 10000, 10250, 10500]

# Create empty lists to store results
moneyness = []
implied_vols = []
timestamps = []
underlying_prices = []
strike_prices = []  # Add this to track which strike each point belongs to

print("Processing data for day 0...")
# Process data for each timestamp
for timestamp in df['timestamp'].unique():
    # Get volcanic rock price
    print(f"Processing timestamp: {timestamp}, day 0")
    # TTE = 8/365 - (timestamp / 1000000) / 365
    # print(f"TTE: {TTE}")
    rock_data = df[(df['timestamp'] == timestamp) & (df['product'] == 'VOLCANIC_ROCK')]
    if not rock_data.empty:
        # Calculate mid price
        S = rock_data['mid_price'].iloc[0]  # Use mid price directly
        
        # Process each voucher strike
        for K in strikes:
            voucher_symbol = f'VOLCANIC_ROCK_VOUCHER_{K}'
            voucher_data = df[(df['timestamp'] == timestamp) & (df['product'] == voucher_symbol)]
            
            if not voucher_data.empty:
                # Calculate mid price of voucher
                V = voucher_data['mid_price'].iloc[0]  # Use mid price directly
                
                # Calculate moneyness
                m = np.log(K/S) / np.sqrt(TTE)
                
                # Calculate implied volatility
                iv = implied_volatility(V, S, K, TTE, r)
                
                # Store results
                if iv > 0.11:
                    moneyness.append(m)
                    implied_vols.append(iv)
                    timestamps.append(timestamp)
                    underlying_prices.append(S)
                    strike_prices.append(K)  # Store the strike price

print("Processing data for day 1...")
df = pd.read_csv('round-3-island-data-bottle/prices_round_3_day_1.csv', sep=';')
TTE = 7 / 365  # 5 days remaining
for timestamp in df['timestamp'].unique():
    # Get volcanic rock price
    print(f"Processing timestamp: {timestamp}, day 1")
    # TTE = 7/365 - (timestamp / 1000000) / 365
    rock_data = df[(df['timestamp'] == timestamp) & (df['product'] == 'VOLCANIC_ROCK')]
    if not rock_data.empty:
        # Calculate mid price
        S = rock_data['mid_price'].iloc[0]  # Use mid price directly
        
        # Process each voucher strike
        for K in strikes:
            voucher_symbol = f'VOLCANIC_ROCK_VOUCHER_{K}'
            voucher_data = df[(df['timestamp'] == timestamp) & (df['product'] == voucher_symbol)]
            
            if not voucher_data.empty:
                # Calculate mid price of voucher
                V = voucher_data['mid_price'].iloc[0]  # Use mid price directly
                
                # Calculate moneyness
                m = np.log(K/S) / np.sqrt(TTE)
                
                # Calculate implied volatility
                iv = implied_volatility(V, S, K, TTE, r)
                
                # Store results
                if iv > 0.11:
                    moneyness.append(m)
                    implied_vols.append(iv)
                    timestamps.append(timestamp)
                    underlying_prices.append(S)
                    strike_prices.append(K)  # Store the strike price

print("Processing data for day 2...")
df = pd.read_csv('round-3-island-data-bottle/prices_round_3_day_2.csv', sep=';')
TTE = 6 / 365  # 5 days remaining
for timestamp in df['timestamp'].unique():
    # Get volcanic rock price
    print(f"Processing timestamp: {timestamp}, day 2")
    # TTE = 6/365 - (timestamp / 1000000) / 365
    rock_data = df[(df['timestamp'] == timestamp) & (df['product'] == 'VOLCANIC_ROCK')]
    if not rock_data.empty:
        # Calculate mid price
        S = rock_data['mid_price'].iloc[0]  # Use mid price directly
        
        # Process each voucher strike
        for K in strikes:
            voucher_symbol = f'VOLCANIC_ROCK_VOUCHER_{K}'
            voucher_data = df[(df['timestamp'] == timestamp) & (df['product'] == voucher_symbol)]
            
            if not voucher_data.empty:
                # Calculate mid price of voucher
                V = voucher_data['mid_price'].iloc[0]  # Use mid price directly
                
                # Calculate moneyness
                m = np.log(K/S) / np.sqrt(TTE)
                
                # Calculate implied volatility
                iv = implied_volatility(V, S, K, TTE, r)
                
                # Store results
                if iv > 0.11:
                    moneyness.append(m)
                    implied_vols.append(iv)
                    timestamps.append(timestamp)
                    underlying_prices.append(S)
                    strike_prices.append(K)  # Store the strike price

# Create DataFrame with results
results = pd.DataFrame({
    'Timestamp': timestamps,
    'Moneyness': moneyness,
    'ImpliedVol': implied_vols,
    'UnderlyingPrice': underlying_prices,
    'Strike': strike_prices
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

# Define colors for each strike
colors = {
    9500: 'blue',
    9750: 'green',
    10000: 'red',
    10250: 'purple',
    10500: 'orange'
}

# Plot each strike with its own color
for strike in strikes:
    strike_data = results[results['Strike'] == strike]
    plt.scatter(strike_data['Moneyness'], strike_data['ImpliedVol'], 
                color=colors[strike], alpha=0.6, label=f'Strike: {strike}')

# Plot the fitted curve
plt.plot(x_fit, y_fit, 'k-', linewidth=2, 
         label=f'Fitted Curve: {params[0]:.4f}x² + {params[1]:.4f}x + {params[2]:.4f}')

# Base IV (at-the-money)
base_iv = parabola(0, *params)
plt.axhline(y=base_iv, color='k', linestyle='--', linewidth=1.5, 
            label=f'Base IV (ATM): {base_iv:.4f}')
plt.axvline(x=0, color='k', linestyle='--', alpha=0.3)

plt.xlabel('Moneyness (m = log(K/S)/sqrt(T))')
plt.ylabel('Implied Volatility')
plt.title('Volatility Smile for Volcanic Rock Vouchers')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Time series of base IV
plt.figure(figsize=(12, 6))

# Group by timestamp and fit a parabola for each timestamp
timestamps = results['Timestamp'].unique()

base_ivs = []
time_points = []

for timestamp in timestamps:
    ts_data = results[results['Timestamp'] == timestamp]
    if len(ts_data) >= 3:  # Need at least 3 points to fit a parabola
        try:
            params, _ = curve_fit(parabola, ts_data['Moneyness'], ts_data['ImpliedVol'])
            base_iv = parabola(0, *params)
            base_ivs.append(base_iv)
            time_points.append(timestamp)
        except:
            continue

# Plot the time series of base IV
plt.plot(time_points, base_ivs, 'b-', label='Base IV (ATM)')
plt.xlabel('Timestamp')
plt.ylabel('Base Implied Volatility')
plt.title('Time Series of Base Implied Volatility')
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()