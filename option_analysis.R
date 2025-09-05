# ----------------------------
# Option Valuation Analysis
# Using Historical Volatility
# ----------------------------

print("Starting script...")
print(paste("Working directory is:", getwd()))

# Load required libraries
library(tidyverse)
library(readr)
library(dplyr)
library(ggplot2)
library(stats)

# ----------------------------
# 1. Load Data
# ----------------------------

# Read stock price history
stock_data <- read_csv("stock_prices.csv")

# Read option snapshot
option_data <- read_csv("option_snapshot.csv")

# ----------------------------
# 2. Calculate Historical Voslatility
# ----------------------------

# Calculate daily log returns
stock_data <- stock_data %>%
  arrange(Date) %>%
  mutate(log_return = log(Close / lag(Close))) %>%
  filter(!is.na(log_return))

# Remove NA from first row
returns <- na.omit(stock_data$log_return)

# Daily volatility
daily_vol <- sd(returns)

# Annualized volatility
annual_vol <- daily_vol * sqrt(252)

cat("Annualized Historical Volatility (σ):", round(annual_vol, 4), "\n")

# ----------------------------
# 3. Black-Scholes Formula (Call Option)
# ----------------------------

black_scholes_call <- function(S, K, T, r, sigma) {
  d1 <- (log(S / K) + (r + 0.5 * sigma^2) * T) / (sigma * sqrt(T))
  d2 <- d1 - sigma * sqrt(T)
  C <- S * pnorm(d1) - K * exp(-r * T) * pnorm(d2)
  return(C)
}

# ----------------------------
# 4. Run Model for This Option
# ----------------------------

# Extract option info
S <- option_data$StockPrice
K <- option_data$Strike
C_market <- option_data$OptionMarketPrice
r <- option_data$RiskFreeRate

# Time to expiration in years
T <- as.numeric(as.Date(option_data$ExpirationDate) - as.Date(option_data$Date)) / 252

# Calculate theoretical price
C_theoretical <- black_scholes_call(S, K, T, r, annual_vol)

# ----------------------------
# 5. Results
# ----------------------------

cat("Market Price of Call Option:", round(C_market, 2), "\n")
cat("Theoretical Price (Black-Scholes):", round(C_theoretical, 2), "\n")

if (C_theoretical > C_market) {
  cat("→ The option was UNDERVALUED based on historical volatility.\n")
} else if (C_theoretical < C_market) {
  cat("→ The option was OVERVALUED based on historical volatility.\n")
} else {
  cat("→ The option was FAIRLY PRICED.\n")
}

# ----------------------------
# 6. Optional: Plot Daily Returns
# ----------------------------

ggplot(stock_data, aes(x = Date, y = log_return)) +
  geom_line(color = "steelblue") +
  labs(title = "Daily Log Returns", y = "Log Return", x = "Date") +
  theme_minimal()
