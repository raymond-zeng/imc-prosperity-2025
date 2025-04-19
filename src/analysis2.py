# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# from scipy.stats import norm
# from scipy.optimize import newton, brentq
# from scipy.interpolate import CubicSpline

# initial_days_to_expiry = 8
# total_steps_per_day    = 1_000_000

# df0 = pd.read_csv('prices_round_3_day_0.csv', sep=';')
# df1 = pd.read_csv('prices_round_3_day_1.csv', sep=';')
# df2 = pd.read_csv('prices_round_3_day_2.csv', sep=';')
# df3 = pd.read_csv('prices_round_4_day_3.csv', sep=';')

# def black_scholes_call(S, K, T, sigma):
#     """
#     Calculate the Black-Scholes option price.
#     S: Current stock price
#     K: Option strike price
#     T: Time to expiration in years
#     sigma: Volatility of the underlying stock
#     """
#     d1 = (np.log(S / K) + (0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
#     d2 = d1 - sigma * np.sqrt(T)
#     return S * norm.cdf(d1) - K * norm.cdf(d2)

# def black_scholes_put(S, K, T, sigma):
#     """
#     Calculate the Black-Scholes option price for a put option.
#     S: Current stock price
#     K: Option strike price
#     T: Time to expiration in years
#     sigma: Volatility of the underlying stock
#     """
#     d1 = (np.log(S / K) + (0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
#     d2 = d1 - sigma * np.sqrt(T)
#     return K * norm.cdf(-d2) - S * norm.cdf(-d1)

# def vega(S, K, T, sigma):
#     """
#     Calculate the Vega of the option.
#     S: Current stock price
#     K: Option strike price
#     T: Time to expiration in years
#     sigma: Volatility of the underlying stock
#     """
#     d1 = (np.log(S / K) + (0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
#     return S * norm.pdf(d1) * np.sqrt(T)

# def implied_volatility(V, S, K, T):
#     """
#     Calculate the implied volatility using the Black-Scholes formula.
#     V: Option market price
#     S: Current stock price
#     K: Option strike price
#     T: Time to expiration in years
#     """
#     # bisection method
#     def objective_function(sigma):
#         if K < S:
#             P = V + K - S
#             return black_scholes_put(S, K, T, sigma) - P
#         return black_scholes_call(S, K, T, sigma) - V
#     vol_low = 0.0001
#     vol_high = 1.0
#     vol_mid = (vol_low + vol_high) / 2.0
#     while vol_high - vol_low > 1e-6:
#         if objective_function(vol_mid) > 0:
#             vol_high = vol_mid
#         else:
#             vol_low = vol_mid
#         vol_mid = (vol_low + vol_high) / 2.0
#     return vol_mid

# def drop(df):
#     df.drop(df[df['product'] == 'RAINFOREST_RESIN'].index, inplace = True)
#     df.drop(df[df['product'] == 'KELP'].index, inplace = True)
#     df.drop(df[df['product'] == 'SQUID_INK'].index, inplace = True)
#     df.drop(df[df['product'] == 'CROISSANTS'].index, inplace = True)
#     df.drop(df[df['product'] == 'JAMS'].index, inplace = True)
#     df.drop(df[df['product'] == 'DJEMBES'].index, inplace = True)
#     df.drop(df[df['product'] == 'PICNIC_BASKET1'].index, inplace = True)
#     df.drop(df[df['product'] == 'PICNIC_BASKET2'].index, inplace = True)
#     df = df.drop(columns = ['bid_price_1', 'bid_price_2', 'bid_price_3', 'ask_price_1', 'ask_price_2', 'ask_price_3', 
#                             'bid_volume_1', 'bid_volume_2', 'bid_volume_3', 'ask_volume_1', 'ask_volume_2', 'ask_volume_3', 'profit_and_loss'])
#     return df
# df0 = drop(df0)
# df1 = drop(df1)
# df2 = drop(df2)
# df3 = drop(df3)
# df3.drop(df3[df3['product'] == 'MAGNIFICENT_MACARONS'].index, inplace = True)

# df1['timestamp'] += 1_000_000
# df2['timestamp'] += 2_000_000
# df3['timestamp'] += 3_000_000
# df = pd.concat([df0, df1, df2, df3], ignore_index=True)

# spot = df[df['product']=='VOLCANIC_ROCK'][['timestamp','mid_price']].rename(columns={'mid_price':'spot'})
# opts = df[df['product'].str.startswith('VOLCANIC_ROCK_VOUCHER')].copy()
# opts['strike'] = opts['product'].str.extract(r'VOLCANIC_ROCK_VOUCHER_(\d+)').astype(float)
# opts['tte'] = 8 / 365 - opts['timestamp'] / 1_000_000 / 365

# merged = pd.merge(opts, spot, on='timestamp')

# # voucher_9750 = merged[merged['strike'] == 9750]
# # voucher_10000 = merged[merged['strike'] == 10000]
# # voucher_10250 = merged[merged['strike'] == 10250]
# # voucher_10500 = merged[merged['strike'] == 10500]
# # voucher_10750 = merged[merged['strike'] == 10750]

# def moneyness(row):
#     return np.log(row['strike'] / row['spot']) / np.sqrt(row['tte'])
# merged = merged[ merged['mid_price'] > 0 ]
# merged['moneyness'] = merged.apply(lambda row: moneyness(row), axis=1)
# merged['implied_vol'] = merged.apply(lambda row: implied_volatility(row['mid_price'], row['spot'], row['strike'], row['tte']), axis=1)
# print(merged)
# merged.to_csv('options_analysis.csv', sep =',', index = False)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv('options_analysis.csv', sep =',')
df = df[df['mid_price'] > 0]
df = df[df['implied_vol'] > 0.05]

#fit quadratic curve to moneyness vs implied volatility
coeffs = np.polyfit(df['moneyness'], df['implied_vol'], deg=2)
x_fit = np.linspace(df['moneyness'].min(), df['moneyness'].max(), 100)
y_fit = np.polyval(coeffs, x_fit)

#graph curve and data points
plt.scatter(df['moneyness'], df['implied_vol'], label='Data Points', color='blue')
plt.plot(x_fit, y_fit, label='Fitted Curve', color='red')
plt.xlabel('Moneyness')
plt.ylabel('Implied Volatility')
plt.title('Moneyness vs Implied Volatility')
plt.legend()
plt.grid()
plt.show()

print(coeffs)