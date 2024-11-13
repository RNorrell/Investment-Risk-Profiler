import os
import pandas as pd
import streamlit as st
import yfinance as yf
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import numpy as np

# Function to fetch stock data from Yahoo Finance (including SPY for S&P 500)
def fetch_stock_data_yahoo(tickers, start_date, end_date):
    stock_data = yf.download(tickers, start=start_date, end=end_date)['Adj Close']
    return stock_data

# Fetch SPY (S&P 500 ETF) data using Yahoo Finance
def fetch_spy_data(start_date, end_date):
    sp500_data = yf.download('SPY', start=start_date, end=end_date)
    return sp500_data

# Function to calculate daily returns
def calculate_daily_returns(data):
    return data.pct_change().dropna()

# Function to calculate volatility (standard deviation of returns)
def calculate_volatility(returns):
    return returns.std() * (252 ** 0.5)  # Annualized volatility

# Function to calculate beta (relative to market returns)
def calculate_beta(stock_returns, market_returns):
    # Combine stock and market returns into a DataFrame
    combined_returns = pd.concat([stock_returns, market_returns], axis=1)
    combined_returns.columns = ['Stock', 'Market']
    
    # Calculate the covariance matrix
    cov_matrix = combined_returns.cov()
    
    # Calculate Beta: covariance between stock and market / variance of market
    beta = cov_matrix.loc['Stock', 'Market'] / cov_matrix.loc['Market', 'Market']
    return beta

# Function to calculate Sharpe ratio (excess return per unit of risk)
def calculate_sharpe_ratio(returns, risk_free_rate=0.02):
    excess_return = returns.mean() - risk_free_rate / 252
    return excess_return / returns.std() * (252 ** 0.5)

# Function to perform risk analysis on the portfolio
def analyze_portfolio(stock_data, tickers):
    stock_returns = calculate_daily_returns(stock_data)
    market_returns = stock_returns['SPY']  # Use SPY as the market index

    risk_metrics = {}
    for ticker in tickers:
        stock_ret = stock_returns[ticker]
        risk_metrics[ticker] = {
            "Volatility": calculate_volatility(stock_ret),
            "Beta": calculate_beta(stock_ret, market_returns),
            "Sharpe Ratio": calculate_sharpe_ratio(stock_ret)
        }

    return pd.DataFrame(risk_metrics)

# Function to plot Risk vs Return (Volatility vs Sharpe Ratio)
# Function to plot Risk vs Return (Volatility vs Sharpe Ratio)
def plot_risk_return(risk_metrics_df):
    fig, ax = plt.subplots(figsize=(10, 6))  # Create a figure and axis object
    ax.scatter(risk_metrics_df.loc['Volatility'], risk_metrics_df.loc['Sharpe Ratio'], color='b')
    
    for ticker in risk_metrics_df.columns:
        ax.text(risk_metrics_df.loc['Volatility'][ticker], risk_metrics_df.loc['Sharpe Ratio'][ticker], ticker)
    
    ax.set_title('Volatility vs Sharpe Ratio')
    ax.set_xlabel('Volatility (Risk)')
    ax.set_ylabel('Sharpe Ratio (Return/Risk)')
    ax.grid(True)
    
    # Display the plot in Streamlit using the updated method
    st.pyplot(fig)


# Portfolio optimization using Efficient Frontier (optional)
def optimize_portfolio(returns):
    num_assets = len(returns.columns)
    cov_matrix = returns.cov()
    
    # Initial equal weight
    weights = num_assets * [1.0 / num_assets]
    
    def portfolio_variance(weights):
        return np.dot(weights, np.dot(cov_matrix, weights))
    
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bounds = tuple((0, 1) for asset in range(num_assets))
    
    result = minimize(portfolio_variance, weights, method='SLSQP', bounds=bounds, constraints=constraints)
    return result.x

# Streamlit UI
def main():
    st.title("Investment Risk Profiler and Diversification Tool")
    
    # User input for stock tickers and dates
    tickers_input = st.text_input("Enter stock tickers (comma-separated)", value="AAPL, MSFT, GOOGL")
    user_tickers = [ticker.strip() for ticker in tickers_input.split(',')]
    
    # Ensure SPY is always included
    tickers = list(set(user_tickers + ['SPY']))  # Add SPY to tickers list and ensure no duplicates
    
    start_date = st.text_input("Start Date", value="2020-01-01")
    end_date = st.text_input("End Date", value="2024-01-01")
    
    # Fetch stock data from Yahoo Finance
    stock_data_yahoo = fetch_stock_data_yahoo(tickers, start_date, end_date)

    # Display fetched stock data in Streamlit
    st.write("Stock Data from Yahoo Finance:", stock_data_yahoo.head())

    # Perform portfolio risk analysis
    risk_metrics_df = analyze_portfolio(stock_data_yahoo, user_tickers)  # Only analyze user tickers
    st.write("Risk Metrics:", risk_metrics_df)

    # Plot risk vs return
    plot_risk_return(risk_metrics_df)

    # Optimize the portfolio and display optimized weights in Streamlit UI
    optimized_weights = optimize_portfolio(calculate_daily_returns(stock_data_yahoo[user_tickers]))
    st.write("Optimized Portfolio Weights:", optimized_weights)

# Run the Streamlit app
if __name__ == "__main__":
    main()
