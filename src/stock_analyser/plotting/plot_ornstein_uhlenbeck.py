import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf

import os
from datetime import datetime


# Download historical data for the commodity (e.g., oil)
ticker = 'NVDA'  # Yahoo Finance symbol for WTI crude oil
start_date = '2010-01-01'
end_date = datetime.now().strftime('%Y-%m-%d')
data = yf.download(ticker, start=start_date, end=end_date)

print(data)



# Calculate daily returns
data['Returns'] = data['Close'].pct_change().dropna()

# Define parameters of the Ornstein-Uhlenbeck model
mu = np.mean(data['Returns'])  # Mean of returns
sigma = np.std(data['Returns'])  # Standard deviation of returns
theta = 0.1  # Mean-reversion parameter
dt = 1/252  # Time step (business days)

# Generate paths of the Ornstein-Uhlenbeck model
num_paths = 10
num_steps = len(data)
paths = np.zeros((num_paths, num_steps))
for i in range(num_paths):
    paths[i, 0] = np.random.normal(0, sigma)
    for t in range(1, num_steps):
        paths[i, t] = paths[i, t-1] + theta * (mu - paths[i, t-1]) * dt + sigma * np.sqrt(dt) * np.random.normal()

# Plot paths of the OU model
plt.figure(figsize=(10, 6))
plt.plot(data.index, paths.T)
plt.title('Ornstein-Uhlenbeck Model for Commodity Volatility')
plt.xlabel('Date')
plt.ylabel('Volatility')
plt.legend(['Path ' + str(i+1) for i in range(num_paths)])

# Save figure
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
filename = f'plots/ornstein_uhlenbeck/{ticker}_ornstein-uhlenbeck_{timestamp}.png'
plt.savefig(filename, bbox_inches='tight')
plt.close()

print(f"Saved {ticker} ornstein-uhlenbeck plot to {filename}")

