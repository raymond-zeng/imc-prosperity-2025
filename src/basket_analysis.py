import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

prices_neg1 = pd.read_csv('prices_round_2_day_-1.csv')
prices_zero = pd.read_csv('prices_round_2_day_0.csv')
prices_one = pd.read_csv('prices_round_2_day_1.csv')

def drop(df):
    df.drop(df[df['product'] == 'SQUID_INK'].index, inplace=True)
    df.drop(df[df['product'] == 'KELP'].index, inplace=True)
    df.drop(df[df['product'] == 'RAINFOREST_RESIN'].index, inplace=True)
    df.drop(columns=['bid_price_1', 'bid_price_2', 'bid_price_3', 'ask_price_1', 'ask_price_2', 'ask_price_3', 'bid_volume_1', 'bid_volume_2', 'bid_volume_3',
                           'ask_volume_1', 'ask_volume_2', 'ask_volume_3'], inplace=True)
    return df

drop(prices_neg1)
drop(prices_zero)
drop(prices_one)

prices_zero['timestamp'] += 1000000
prices_one['timestamp'] += 2000000

neg1_croissants = prices_neg1[prices_neg1['product'] == 'CROISSANTS']
neg1_jam = prices_neg1[prices_neg1['product'] == 'JAMS']
neg1_djembe = prices_neg1[prices_neg1['product'] == 'DJEMBES']
neg1_picnic_basket1 = prices_neg1[prices_neg1['product'] == 'PICNIC_BASKET1']
neg1_picnic_basket2 = prices_neg1[prices_neg1['product'] == 'PICNIC_BASKET2']

zero_croissants = prices_zero[prices_zero['product'] == 'CROISSANTS']
zero_jam = prices_zero[prices_zero['product'] == 'JAMS']
zero_djembe = prices_zero[prices_zero['product'] == 'DJEMBES']
zero_picnic_basket1 = prices_zero[prices_zero['product'] == 'PICNIC_BASKET1']
zero_picnic_basket2 = prices_zero[prices_zero['product'] == 'PICNIC_BASKET2']

one_croissants = prices_one[prices_one['product'] == 'CROISSANTS']
one_jam = prices_one[prices_one['product'] == 'JAMS']
one_djembe = prices_one[prices_one['product'] == 'DJEMBES']
one_picnic_basket1 = prices_one[prices_one['product'] == 'PICNIC_BASKET1']
one_picnic_basket2 = prices_one[prices_one['product'] == 'PICNIC_BASKET2']

croissants = pd.concat([neg1_croissants, zero_croissants, one_croissants])
jam = pd.concat([neg1_jam, zero_jam, one_jam])
djembe = pd.concat([neg1_djembe, zero_djembe, one_djembe])
picnic_basket1 = pd.concat([neg1_picnic_basket1, zero_picnic_basket1, one_picnic_basket1])
picnic_basket2 = pd.concat([neg1_picnic_basket2, zero_picnic_basket2, one_picnic_basket2])

croissants = croissants[['timestamp', 'mid_price']]
jam = jam[['timestamp', 'mid_price']]
djembe = djembe[['timestamp', 'mid_price']]
picnic_basket2 = picnic_basket2[['timestamp', 'mid_price']]
picnic_basket1 = picnic_basket1[['timestamp', 'mid_price']]

picnic_basket1.rename(columns={'mid_price': 'picnic_basket1_mid_price'}, inplace=True)
picnic_basket2.rename(columns={'mid_price': 'picnic_basket2_mid_price'}, inplace=True)
croissants.rename(columns={'mid_price': 'croissants_mid_price'}, inplace=True)
jam.rename(columns={'mid_price': 'jam_mid_price'}, inplace=True)
djembe.rename(columns={'mid_price': 'djembe_mid_price'}, inplace=True)

merged = pd.merge(croissants, picnic_basket2, on='timestamp')
merged = pd.merge(merged, jam, on='timestamp')
merged = pd.merge(merged, picnic_basket1, on='timestamp')
merged = pd.merge(merged, djembe, on='timestamp')
merged['picnic_basket1_spread'] = merged['picnic_basket1_mid_price'] - 6 * merged['croissants_mid_price'] - 3 * merged['jam_mid_price'] - merged['djembe_mid_price']
merged['picnic_basket2_spread'] = merged['picnic_basket2_mid_price'] - 4 * merged['croissants_mid_price'] - 2 * merged['jam_mid_price']

# plot the new mid price of picnic basket 2 over time and the mid price of croissants over time
plt.figure(figsize=(12, 6))
plt.title('Spread of Picnic Basket 1 Over Time')
plt.xlabel('Time')
plt.ylabel('Spread')
plt.xticks(rotation=45)
plt.plot(merged['timestamp'], merged['picnic_basket1_spread'], label='Picnic Basket 1')
plt.legend()
plt.show()