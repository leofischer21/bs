# option_simulation_analysis.R

library(tidyverse)
library(lubridate)

set.seed(42)

# --- 1. Simuliere Aktienkurse (60 Handelstage) ---
dates <- seq.Date(from = as.Date("2024-01-01"), by = "day", length.out = 90)
# Nur Wochentage (Handelstage)
dates <- dates[weekdays(dates) %in% c("Monday", "Tuesday", "Wednesday", "Thursday", "Friday")]
n <- length(dates)

# Starte bei 100, tägliche Rendite ~ N(0, 0.01)
returns <- rnorm(n, mean = 0, sd = 0.01)
prices <- 100 * cumprod(1 + returns)

stock_data <- tibble(Date = dates, Close = prices)

# --- 2. Black-Scholes Call Option Preis Funktion ---
bs_call_price <- function(S, K, T, r, sigma) {
  d1 <- (log(S / K) + (r + sigma^2 / 2) * T) / (sigma * sqrt(T))
  d2 <- d1 - sigma * sqrt(T)
  C <- S * pnorm(d1) - K * exp(-r * T) * pnorm(d2)
  return(C)
}

# --- 3. Funktion zur historischen Volatilität (annualisiert) ---
calc_hist_vol <- function(prices) {
  if(length(prices) < 2) return(NA_real_)
  log_returns <- diff(log(prices))
  sd(log_returns) * sqrt(252)
}

# --- 4. Simuliere Optionsdaten ---
strike_price <- 102
risk_free_rate <- 0.02
# Verwende ein festes Verfallsdatum
fixed_expiration_date <- as.Date("2024-03-29") 

option_data <- tibble(
  Date = dates,
  StockPrice = prices,
  Strike = strike_price,
  RiskFreeRate = risk_free_rate
)

option_data <- option_data %>%
  mutate(
    # Korrekte Berechnung der verbleibenden Laufzeit bis zum festen Verfallsdatum
    TimeToExp = as.numeric(fixed_expiration_date - Date) / 365,
    HistVol = map_dbl(row_number(), function(i) {
      start_idx <- max(1, i - 29)
      calc_hist_vol(stock_data$Close[start_idx:i])
    }),
    TheoreticalPrice = mapply(bs_call_price, StockPrice, Strike, TimeToExp, RiskFreeRate, HistVol)
  )

# --- 5. Simuliere Marktpreis der Option mit ±10% Noise ---
n_options <- nrow(option_data)
option_data <- option_data %>%
  mutate(
    OptionMarketPrice = TheoreticalPrice * runif(n_options, 0.9, 1.1)
  )

# --- 6. Berechne tägliche Log Returns ---
stock_data <- stock_data %>%
  arrange(Date) %>%
  mutate(log_return = log(Close / lag(Close)))

# --- 7. Ergebnisübersicht ---
cat("Simulierte Optionsdaten (erste 6 Zeilen):\n")
print(head(option_data))

# --- 8. Plots ---
library(ggplot2)

# Log Returns Plot
p1 <- ggplot(stock_data %>% filter(!is.na(log_return)), aes(x = Date, y = log_return)) +
  geom_line(color = "steelblue") +
  labs(title = "Simulierte tägliche Log Returns", y = "Log Return", x = "Datum") +
  theme_minimal()

# Optionspreise Plot
p2 <- ggplot(option_data %>% filter(!is.na(TheoreticalPrice), !is.na(OptionMarketPrice)), aes(x = Date)) +
  geom_line(aes(y = TheoreticalPrice, color = "Theoretischer Preis")) +
  geom_line(aes(y = OptionMarketPrice, color = "Marktpreis")) +
  labs(title = "Option Preis: Theoretisch vs Markt", y = "Preis", x = "Datum", color = "Legende") +
  theme_minimal()

# Plots speichern - die print() Befehle sind nicht nötig
ggsave("plot_log_returns.png", p1, width = 7, height = 5)
ggsave("plot_option_prices.png", p2, width = 7, height = 5)
