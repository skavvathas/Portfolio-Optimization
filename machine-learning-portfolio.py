#!/usr/bin/env python
# coding: utf-8

# # Portfolio Optimizattion

# Develop a portfolio optimization model implementing key financial strategies: Mean-Variance Optimization,
# Max Sharpe Ratio, Minimum Variance, Equal Weight, and Hierarchical Risk Parity (HRP).

# In[1]:


import yfinance as yf
import pandas as pd
import numpy as np
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import squareform
from scipy.optimize import minimize
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import squareform
import matplotlib.pyplot as plt

# tickers
tickers = [
    "AAPL", "MSFT", "AMZN", "GOOGL", "META", "BRK-B", "NVDA", "JPM", "JNJ", "V",
    "UNH", "HD", "PG", "DIS", "MA", "PYPL", "BAC", "XOM", "VZ", "ADBE",
    "CMCSA", "NFLX", "KO", "T", "PFE", "MRK", "CSCO", "PEP", "INTC", "ABT",
    "CVX", "ORCL", "WMT", "CRM", "ACN", "COST", "NKE", "TXN", "QCOM", "MCD",
    "MDT", "NEE", "HON", "LIN", "IBM", "SBUX", "LOW", "AMD", "GE", "UPS"
]

# Download data grouped by ticker
data = yf.download(tickers, start="2022-01-01", end="2025-01-01")

data


# When downloading multiple tickers without group_by='ticker', yfinance returns a MultiIndex DataFrame where the first level is the price type (Open, High, Low, Close, etc.) and the second level is the ticker symbol. So to get the Close prices only:

# In[2]:


close_prices = data['Close']  # Select first level 'Close'
close_prices = close_prices.dropna()  # Drop any rows with missing values
close_prices


# Now close_prices is a DataFrame of Close prices indexed by date and columns as tickers.

# Calculate daily returns

# In[3]:


# Calculate daily returns
returns = close_prices.pct_change().dropna()

# Annualize returns and covariance
mean_returns = returns.mean() * 252
cov_matrix = returns.cov() * 252
num_assets = len(tickers)
risk_free_rate = 0.15

print(close_prices.head())
print(returns.head())


# ### Portfolio metrics functions

# In portfolio optimization, three primary metrics are used to evaluate and compare investment strategies: expected return, volatility, and the Sharpe ratio. 
# 
# The **expected return** represents the average annualized return the portfolio is projected to earn based on historical or forecasted asset returns. 
# 
# **Volatility**, measured as the standard deviation of returns, quantifies the portfolio’s total risk — the higher the volatility, the more uncertain the returns.
# 
# The **Sharpe ratio** combines these two by measuring the risk-adjusted return, i.e., how much excess return is earned per unit of risk above the risk-free rate.

# These metrics are computed using the portfolio weights, mean returns, and the covariance matrix of asset returns. They are then used as objective functions in various optimization models — for example, maximizing the Sharpe ratio, minimizing volatility, or maximizing return for a given risk. Constraints (such as full capital allocation) and bounds (like no short selling) are applied to ensure practical, real-world portfolio solutions. Together, these metrics form the foundation for constructing efficient portfolios aligned with different investor preferences.

# In[4]:


def portfolio_performance(weights, mean_returns, cov_matrix):
    returns = np.dot(weights, mean_returns)
    std = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    return returns, std

def neg_sharpe_ratio(weights, mean_returns, cov_matrix, risk_free_rate):
    p_ret, p_vol = portfolio_performance(weights, mean_returns, cov_matrix)
    return -(p_ret - risk_free_rate) / p_vol

def portfolio_volatility(weights, mean_returns, cov_matrix):
    return portfolio_performance(weights, mean_returns, cov_matrix)[1]

# Constraints and bounds
constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
bounds = tuple((0, 1) for _ in range(num_assets))


# ## Step 3: Optimization techniques

# ### 3.1 Mean-Variance (target return) Optimization - maximize return for given risk

# **Mean-Variance Optimization (MVO)**, introduced by Harry Markowitz, is a foundational concept in modern portfolio theory. It aims to construct an optimal portfolio by selecting asset weights that maximize the expected return for a given level of risk (or equivalently, minimize risk for a given expected return).
# 
# In this specific implementation, the goal is to maximize portfolio return, subject to:
# 
# Full investment constraint (weights sum to 1),
# 
# Bounds on asset weights (e.g., no short-selling if lower bound = 0),
# 
# Additional constraints as needed (e.g., target risk level or diversification requirements).
# 
# The optimization problem is solved using SciPy’s minimize function with the SLSQP (Sequential Least Squares Programming) method.

# In[5]:


def max_return_portfolio():
    # Maximize return under full investment and weights bounds
    def neg_return(weights):
        return -np.dot(weights, mean_returns)
    
    opt = minimize(neg_return, num_assets * [1. / num_assets,], method='SLSQP', bounds=bounds, constraints=constraints)
    return opt.x


# ### 3.2 Max Sharpe Ratio (Tangency Portfolio)

# The **Sharpe Ratio** measures the risk-adjusted return of a portfolio by comparing its excess return (above the risk-free rate) to its volatility. The Tangency Portfolio is the portfolio on the efficient frontier that has the highest Sharpe Ratio, meaning it provides the best return per unit of risk.
# 
# This optimization aims to maximize the Sharpe Ratio, subject to:
# 
# Full investment (weights sum to 1),
# 
# Asset weight bounds (e.g., no short-selling),
# 
# Other optional constraints.
# 
# In this implementation:
# 
# The objective function neg_sharpe_ratio returns the negative Sharpe Ratio so that the minimize function can find the maximum.
# 
# The optimization is solved using SciPy’s minimize with the SLSQP method.
# 
# This approach is popular for constructing portfolios that aim to achieve the most efficient risk-return trade-off possible.

# In[6]:


def max_sharpe_ratio_portfolio():
    opt = minimize(neg_sharpe_ratio, num_assets * [1. / num_assets,], args=(mean_returns, cov_matrix, risk_free_rate),
                   method='SLSQP', bounds=bounds, constraints=constraints)
    return opt.x


# ### 3.3 Minimum Variance Portfolio

# The **Minimum Variance Portfolio (MVP)** is the portfolio that achieves the lowest possible volatility (risk), regardless of expected returns. It lies at the leftmost point of the efficient frontier, making it ideal for highly risk-averse investors.
# 
# This approach minimizes portfolio variance (or standard deviation) under constraints:
# 
# Full investment (weights sum to 1),
# 
# No short-selling or other weight bounds, unless specified.
# 
# In this implementation:
# 
# The objective function portfolio_volatility computes the portfolio's total standard deviation.
# 
# The minimize function from SciPy is used with the SLSQP algorithm, subject to the given constraints and bounds.
# 
# The result is a diversified, low-risk allocation that minimizes the overall portfolio fluctuations — especially valuable during uncertain market conditions.

# In[7]:


def min_variance_portfolio():
    opt = minimize(portfolio_volatility, num_assets * [1. / num_assets,], args=(mean_returns, cov_matrix),
                   method='SLSQP', bounds=bounds, constraints=constraints)
    return opt.x


# ### 3.4 Equal Weight Portfolio

# The Equal Weight Portfolio is the simplest form of asset allocation, where each asset in the portfolio is assigned an equal proportion of the total capital.
# 
# This strategy assumes:
# 
# No preference among assets,
# 
# No need for forecasts or optimizations,
# 
# Each asset contributes equally to the portfolio's exposure.
# 
# In this implementation:
# 
# The weights are computed by assigning 1/N to each of the N assets, resulting in a uniform distribution of capital.
# 
# While this method does not optimize for risk or return, it serves as a useful benchmark against more advanced strategies like Mean-Variance or Maximum Sharpe. It often performs surprisingly well in practice due to its simplicity, diversification, and avoidance of estimation errors.

# In[8]:


def equal_weight_portfolio():
    return np.array(num_assets * [1. / num_assets])


# ### 3.5 Hierarchical Risk Parity (HRP) Portfolio

# The Hierarchical Risk Parity (HRP) portfolio is an advanced asset allocation strategy that combines hierarchical clustering with risk budgeting to distribute capital more robustly across assets.
# 
# Unlike traditional mean-variance optimization, HRP:
# 
# Avoids direct matrix inversion, reducing sensitivity to estimation errors,
# 
# Uses a dendrogram-based clustering to group assets by correlation,
# 
# Allocates capital recursively based on cluster-level variances.
# 
# Key steps in HRP:
# 
# - Compute a distance matrix from the correlation matrix.
# 
# - Perform hierarchical clustering to organize assets by similarity.
# 
# Apply a recursive bisection algorithm to allocate weights:
# 
# Clusters with lower variance receive more capital.
# 
# Allocations are updated in a top-down fashion through the cluster tree.

# In[9]:


def correl_dist(corr):
    """Compute distance matrix from correlation matrix for clustering"""
    dist = np.sqrt(0.5 * (1 - corr))
    return dist

def get_quasi_diag(link):
    """Sort clustered items by distance to get ordering for HRP"""
    link = link.astype(int)
    sort_idx = []

    def recursive_sort(node):
        if node < len(cov_matrix):
            sort_idx.append(node)
        else:
            recursive_sort(link[node - len(cov_matrix), 0])
            recursive_sort(link[node - len(cov_matrix), 1])

    recursive_sort(link.shape[0] + len(cov_matrix) - 2)
    return sort_idx

def get_cluster_var(cov, cluster_items):
    """Calculate variance per cluster"""
    cov_slice = cov.loc[cluster_items, cluster_items]
    weights = np.linalg.inv(cov_slice).sum(axis=1)
    weights /= weights.sum()
    var = np.dot(weights.T, np.dot(cov_slice, weights))
    return var

def hrp_allocation(cov, corr):
    dist = correl_dist(corr)
    link = linkage(squareform(dist), 'single')
    sort_ix = get_quasi_diag(link)
    sorted_tickers = cov.index[sort_ix].tolist()

    hrp = pd.Series(1, index=sorted_tickers)

    def recursive_bipartition(items):
        if len(items) == 1:
            return
        split = len(items) // 2
        cluster1 = items[:split]
        cluster2 = items[split:]
        var1 = get_cluster_var(cov, cluster1)
        var2 = get_cluster_var(cov, cluster2)
        alpha = 1 - var1 / (var1 + var2)
        recursive_bipartition(cluster1)
        recursive_bipartition(cluster2)
        for i in cluster1:
            hrp[i] *= alpha
        for i in cluster2:
            hrp[i] *= (1 - alpha)

    recursive_bipartition(sorted_tickers)
    hrp /= hrp.sum()
    # Reorder back to original order
    return hrp.reindex(cov.index).fillna(0).values


# ### Calculate HRP weights

# In[10]:


corr_matrix = returns.corr()
hrp_weights = hrp_allocation(cov_matrix, corr_matrix)


# ### Calculate weights for all portfolios

# In[11]:


weights_max_ret = max_return_portfolio()
weights_max_ret


# In[12]:


weights_max_sharpe = max_sharpe_ratio_portfolio()
weights_max_sharpe


# In[13]:


weights_min_var = min_variance_portfolio()
weights_min_var


# In[14]:


weights_equal = equal_weight_portfolio()
weights_equal


# ### Show results

# In[15]:


def display_portfolio(name, weights):
    ret, vol = portfolio_performance(weights, mean_returns, cov_matrix)
    sharpe = (ret - risk_free_rate) / vol
    print(f"{name} Portfolio:")
    print(f"  Expected Annual Return: {ret:.2%}")
    print(f"  Annual Volatility: {vol:.2%}")
    print(f"  Sharpe Ratio: {sharpe:.2f}")
    print()

display_portfolio("Max Return", weights_max_ret)
display_portfolio("Max Sharpe Ratio", weights_max_sharpe)
display_portfolio("Min Variance", weights_min_var)
display_portfolio("Equal Weight", weights_equal)
display_portfolio("Hierarchical Risk Parity (HRP)", hrp_weights)


# ### 🔍 Portfolio-by-Portfolio Explanation:
# 
# ## ✅ Max Return Portfolio
# 
# Return: 65.22%
# 
# Volatility: 55.27%
# 
# Sharpe Ratio: 0.91
# 
# 🔸 This strategy goes all-in on the highest-returning asset(s), regardless of risk.
# 
# 🔸 It offers a very high return, but at the cost of high volatility (risk).
# 
# 🔸 The Sharpe ratio is decent, showing that the return justifies the risk to some extent — but it’s still quite aggressive.
# 
# ### ⭐ Max Sharpe Ratio Portfolio
# 
# Return: 39.49%
# 
# Volatility: 22.71%
# 
# Sharpe Ratio: 1.08
# 
# 🔸 This portfolio aims to maximize return per unit of risk, resulting in the highest Sharpe ratio here.
# 
# 🔸 It offers a strong return but with much lower volatility than the max return portfolio.
# 
# 🔸 Often considered the most efficient portfolio in traditional finance.
# 
# ### 🛡️ Minimum Variance Portfolio
# 
# Return: 9.76%
# 
# Volatility: 11.49%
# 
# Sharpe Ratio: -0.46
# 
# 🔸 Focuses on minimizing volatility, without regard to expected return.
# 
# 🔸 Very conservative — lowest risk, but return is low and underperforms the risk-free rate (hence, negative Sharpe).
# 
# 🔸 Suitable in extremely uncertain markets or for risk-averse investors.
# 
# 
# ### ⚖️ Equal Weight Portfolio
# 
# Return: 8.61%
# 
# Volatility: 16.73%
# 
# Sharpe Ratio: -0.38
# 
# 🔸 Allocates equally across all assets, regardless of return or risk.
# 
# 🔸 Simple and naive but often used as a benchmark.
# 
# 🔸 Again, negative Sharpe ratio suggests underperformance vs. the risk-free rate.
# 
# ### 🧠 Hierarchical Risk Parity (HRP) Portfolio
# 
# Return: 7.57%
# 
# Volatility: 14.17%
# 
# Sharpe Ratio: -0.52
# 
# 🔸 Uses machine learning techniques (clustering) to balance risk across correlated assets.
# 
# 🔸 Relatively moderate risk and return — the goal is robustness, not necessarily high performance.
# 
# 🔸 The negative Sharpe ratio here indicates that returns did not compensate for the risk, likely due to noisy data or market conditions.

# Five portfolio strategies tested, each with distinct objectives and results. 
# 
# Below is a summary of what each method achieved based on expected annual return, volatility (risk), and Sharpe ratio (risk-adjusted return):

# | Portfolio                          | Expected Return | Volatility | Sharpe Ratio | Interpretation                                                          |
# | ---------------------------------- | --------------: | ---------: | -----------: | ----------------------------------------------------------------------- |
# | **Max Return**                     |      **65.22%** |     55.27% |         0.91 | Very high return, but also very risky; suited for aggressive investors. |
# | **Max Sharpe Ratio**               |      **39.49%** |     22.71% |     **1.08** | Best overall trade-off; high return with reasonable risk.               |
# | **Min Variance**                   |           9.76% | **11.49%** |        -0.46 | Safest portfolio, but too conservative with poor returns.               |
# | **Equal Weight**                   |           8.61% |     16.73% |        -0.38 | Simple and diversified baseline; underperforms optimized strategies.    |
# | **Hierarchical Risk Parity (HRP)** |           7.57% |     14.17% |        -0.52 | Most diversified; good from a risk-distribution view, but low return.   |
# 

# ### Conclusion

# - Max Sharpe Ratio is the most efficient portfolio — it provides the best return per unit of risk.
# 
# - HRP offers a modern, machine learning-driven approach that avoids overfitting and doesn’t rely on matrix inversion — helpful in volatile markets.
# 
# - Classical methods like Mean-Variance and Min-Variance remain relevant but must be chosen based on investor goals (growth vs. safety).
# 
# - Equal Weight is simple but suboptimal, highlighting the power of optimization techniques.

# ### Plot weights for comparison

# In[16]:


plt.figure(figsize=(15,7))
width = 0.15
ind = np.arange(num_assets)

plt.bar(ind, weights_equal, width, label='Equal Weight')
plt.bar(ind + width, weights_min_var, width, label='Min Variance')
plt.bar(ind + 2*width, weights_max_sharpe, width, label='Max Sharpe')
plt.bar(ind + 3*width, weights_max_ret, width, label='Max Return')
plt.bar(ind + 4*width, hrp_weights, width, label='HRP')

plt.xticks(ind + 2*width, tickers, rotation=90)
plt.ylabel('Weights')
plt.title('Portfolio Weights Comparison')
plt.legend()
plt.tight_layout()
plt.show()


# In[ ]:




