# Portfolio Optimization in Python

This project implements a comprehensive **portfolio optimization framework** using Python, focusing on multiple investment strategies and modern financial techniques.

## Project Overview

The analysis covers **five portfolio strategies**:

1. **Max Return Portfolio**  
   - Seeks the highest expected return regardless of risk.  
   - Aggressive strategy with high volatility.

2. **Max Sharpe Ratio Portfolio (Tangency Portfolio)**  
   - Optimizes risk-adjusted return, maximizing return per unit of risk.  
   - Provides the most efficient trade-off between return and risk.

3. **Minimum Variance Portfolio (MVP)**  
   - Focuses on minimizing portfolio volatility.  
   - Suitable for risk-averse investors; conservative allocation.

4. **Equal Weight Portfolio**  
   - Allocates equal capital to all assets.  
   - Simple benchmark, diversified but suboptimal.

5. **Hierarchical Risk Parity (HRP) Portfolio**  
   - Advanced strategy using **machine learning (clustering)** to allocate risk across correlated assets.  
   - Avoids matrix inversion sensitivity; robust in volatile markets.

## Key Features

- **📊 Portfolio Metrics:** Calculates expected annualized return, volatility, and Sharpe ratio for each strategy.  
- **🔧 Optimization Methods:** Implements **Mean-Variance Optimization**, **Max Sharpe Ratio**, **Min Variance**, **Equal Weight**, and **HRP**.  
- **📈 Data Handling:** Downloads historical prices for 50 large-cap US equities using **yfinance**.  
- **📉 Risk Management:** Visualizes portfolio weights and compares allocation across strategies.  
- **🧠 Modern Techniques:** HRP leverages hierarchical clustering for robust risk-based allocation.

## Insights

- **Max Sharpe Ratio Portfolio** provides the best return-to-risk trade-off.  
- **HRP** ensures diversification and reduces sensitivity to estimation errors.  
- **Equal Weight** is a simple but useful baseline, often underperforming optimized strategies.  
- Classical methods like **Mean-Variance** and **Min Variance** remain relevant depending on investor objectives.

## Tools & Libraries

- 🐍 **Python** – Core programming  
- 📊 **pandas** – Data handling  
- 🔢 **numpy** – Numerical calculations  
- 📉 **matplotlib** – Visualization  
- 📈 **scipy.optimize** – Optimization routines  
- 🧮 **scipy.cluster.hierarchy** – Hierarchical clustering for HRP  
- 💹 **yfinance** – Historical market data retrieval
