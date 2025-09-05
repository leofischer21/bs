import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm

# --- 1. Funktion zur Aktienkurs-Simulation (Geometric Brownian Motion)
def simulate_stock_data(s0=100, mu=0.1, sigma=0.2, trading_years=5, trading_days=252, seed=42):
    np.random.seed()
    dt = 1 / trading_days
    n = trading_years * trading_days
    returns = np.random.normal(loc=(mu - 0.5 * sigma**2) * dt,
                               scale=sigma * np.sqrt(dt),
                               size=n)
    log_prices = np.log(s0) + np.cumsum(returns)
    prices = np.exp(log_prices)
    dates = pd.date_range(start='2020-01-01', periods=n, freq='B')
    return pd.Series(prices, index=dates)

# --- 2. Black-Scholes-Formel für Call-Option
def black_scholes_call_price(spot, strike, time_to_maturity, rate, volatility, dividend=0.0):
    if time_to_maturity <= 0:
        return max(spot - strike, 0)
    d1 = (np.log(spot / strike) + (rate - dividend + 0.5 * volatility ** 2) * time_to_maturity) / \
         (volatility * np.sqrt(time_to_maturity))
    d2 = d1 - volatility * np.sqrt(time_to_maturity)
    call_price = spot * np.exp(-dividend * time_to_maturity) * norm.cdf(d1) - \
                 strike * np.exp(-rate * time_to_maturity) * norm.cdf(d2)
    return call_price

# --- 3. Simulation & Pricing
stock_prices = simulate_stock_data()
np.random.seed(0)
random_index = np.random.randint(30, len(stock_prices))

# Evaluierungsdatum (Kaufzeitpunkt)
evaluation_date = stock_prices.index[random_index]
spot_price = stock_prices.iloc[random_index]
strike_price = spot_price * 1.05
time_to_expiry = (len(stock_prices) - random_index) / 252

# Parameter
annual_risk_free_rate = 0.03
annual_volatility = 0.2

# Preisberechnung am Kaufdatum
option_price_at_purchase = black_scholes_call_price(
    spot=spot_price,
    strike=strike_price,
    time_to_maturity=time_to_expiry,
    rate=annual_risk_free_rate,
    volatility=annual_volatility
)

# --- 4. Kurs am Verfallstag (am Ende der Simulation)
expiry_date = stock_prices.index[-1]
spot_at_expiry = stock_prices.iloc[-1]
intrinsic_value = max(spot_at_expiry - strike_price, 0)
profit = intrinsic_value - option_price_at_purchase

# --- 5. Plot
plt.figure(figsize=(10, 5))
plt.plot(stock_prices.index, stock_prices.values, label='Stock Price')
plt.axvline(evaluation_date, color='red', linestyle='--', label='Evaluation Date')
plt.axvline(expiry_date, color='green', linestyle='--', label='Expiration Date')
plt.title('Simulierte Aktienkursentwicklung')
plt.xlabel('Datum')
plt.ylabel('Preis')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# --- 6. Ergebnis ausgeben
print("------ Option Pricing Beispiel ------")
print(f"📅 Evaluation Date        : {evaluation_date.date()}")
print(f"💰 Spot Price (S)         : {spot_price:.2f}")
print(f"🎯 Strike Price (K)       : {strike_price:.2f}")
print(f"⏳ Time to Maturity (T)   : {time_to_expiry:.2f} Jahre")
print(f"📈 Risk-Free Rate (r)     : {annual_risk_free_rate}")
print(f"📊 Volatility (σ)         : {annual_volatility}")
print(f"💵 Call Option Price      : {option_price_at_purchase:.2f}")
print()
print("------ Am Ende der Laufzeit ------")
print(f"📆 Expiry Date            : {expiry_date.date()}")
print(f"📉 Spot at Expiry         : {spot_at_expiry:.2f}")
print(f"💸 Intrinsic Value        : {intrinsic_value:.2f}")
print(f"📊 Gewinn / Verlust       : {profit:.2f} €")
