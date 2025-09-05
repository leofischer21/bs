import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm
import streamlit as st

def simulate_gbm(s0, mu, sigma, T, steps, seed=None):
    if seed is not None:
        np.random.seed(seed)
    dt = T / steps
    prices = [s0]
    for _ in range(steps):
        shock = np.random.normal((mu - 0.5 * sigma**2) * dt, sigma * np.sqrt(dt))
        prices.append(prices[-1] * np.exp(shock))
    return prices

def black_scholes_call(s, k, T, r, sigma):
    if T <= 0:
        return max(s - k, 0)
    d1 = (np.log(s / k) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    call_price = s * norm.cdf(d1) - k * np.exp(-r * T) * norm.cdf(d2)
    return call_price

def main():
    st.title("📈 GBM Stock Simulator & Black-Scholes Option Pricing")

    # User inputs
    s0 = st.number_input("Initial Stock Price (S0)", value=100.0, min_value=0.01)
    mu = st.slider("Expected Return (mu)", min_value=-0.5, max_value=0.5, value=0.07, step=0.01)
    sigma = st.slider("Volatility (sigma)", min_value=0.01, max_value=1.0, value=0.2, step=0.01)
    T = st.slider("Time Horizon (Years)", min_value=0.01, max_value=5.0, value=1.0, step=0.01)
    steps = st.slider("Number of Steps", min_value=50, max_value=1000, value=252)
    r = st.slider("Risk-free Rate (r)", min_value=0.0, max_value=0.2, value=0.03, step=0.001)
    K = st.number_input("Strike Price (K)", value=105.0, min_value=0.01)
    seed = st.number_input("Random Seed (leave 0 for random)", value=0, min_value=0, step=1)

    # Seed handling
    actual_seed = None if seed == 0 else int(seed)

    # Simulate
    prices = simulate_gbm(s0, mu, sigma, T, steps, seed=actual_seed)
    times = np.linspace(0, T, steps + 1)

    # Calculate option prices over time
    option_prices = [black_scholes_call(price, K, T - t, r, sigma) for price, t in zip(prices, times)]

    # Display results
    st.write(f"### Simulated stock price path with {steps} steps over {T} years")
    st.line_chart(pd.DataFrame({"Stock Price": prices, "Option Price": option_prices}, index=times))

    st.write("### Final Results:")
    st.write(f"Stock Price at Maturity: {prices[-1]:.2f}")
    st.write(f"Option Price at t=0: {option_prices[0]:.2f}")
    st.write(f"Option Price at Maturity: {option_prices[-1]:.2f}")
    st.write(f"Option Intrinsic Value at Maturity: {max(prices[-1] - K, 0):.2f}")

if __name__ == "__main__":
    main()
