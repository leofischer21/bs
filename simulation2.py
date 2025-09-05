import numpy as np
import pandas as pd
from scipy.stats import norm
import streamlit as st

# --- Simulation Function ---
def simulate_gbm(s0, mu, sigma, T, steps, seed=None):
    if seed is not None:
        np.random.seed(seed)
    dt = T / steps
    prices = [s0]
    for _ in range(steps):
        shock = np.random.normal((mu - 0.5 * sigma**2) * dt, sigma * np.sqrt(dt))
        prices.append(prices[-1] * np.exp(shock))
    return prices

# --- Black-Scholes Pricing Function ---
def black_scholes_call(s, k, T, r, sigma):
    if T <= 0:
        return max(s - k, 0)
    d1 = (np.log(s / k) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    call_price = s * norm.cdf(d1) - k * np.exp(-r * T) * norm.cdf(d2)
    return call_price

# --- Streamlit UI ---
def main():
    st.set_page_config(page_title="📈 GBM + Black-Scholes Option Simulator")
    st.title("📈 GBM Stock Simulator & Black-Scholes Option Pricing")

    # --- Inputs ---
    s0 = st.number_input("Initial Stock Price (S₀)", value=100.0, min_value=0.01)
    mu = st.slider("Expected Return (μ)", -0.5, 0.5, 0.1, 0.01)
    sigma = st.slider("Volatility (σ)", 0.01, 1.0, 0.2, 0.01)
    T = st.slider("Time Horizon (Years)", 0.1, 5.0, 1.0, 0.1)
    steps = st.slider("Number of Steps", 50, 1000, 252, 10)
    r = st.slider("Risk-Free Rate (r)", 0.0, 0.2, 0.03, 0.001)
    K = st.number_input("Strike Price (K)", value=105.0, min_value=0.01)
    seed = st.number_input("Random Seed (optional)", value=0, min_value=0)

    # --- Seed Logic ---
    actual_seed = None if seed == 0 else int(seed)

    # --- Simulate GBM ---
    prices = simulate_gbm(s0, mu, sigma, T, steps, seed=actual_seed)
    times = np.linspace(0, T, steps + 1)

    # --- Option Pricing Over Time ---
    option_prices = [black_scholes_call(S, K, T - t, r, sigma) for S, t in zip(prices, times)]

    # --- Display ---
    st.subheader("📊 Simulated Stock & Option Price Over Time")
    df = pd.DataFrame({
        "Time (Years)": times,
        "Stock Price": prices,
        "Call Option Price": option_prices
    }).set_index("Time (Years)")

    st.line_chart(df)

    # --- Results ---
    st.subheader("📢 Final Results")
    st.write(f"📅 **Time to maturity:** {T:.2f} years")
    st.write(f"💲 **Final Stock Price:** {prices[-1]:.2f}")
    st.write(f"📈 **Call Option Price at t=0:** {option_prices[0]:.2f}")
    st.write(f"📈 **Call Option Price at maturity:** {option_prices[-1]:.2f}")
    st.write(f"💰 **Intrinsic Value at maturity:** {max(prices[-1] - K, 0):.2f}")

if __name__ == "__main__":
    main()
