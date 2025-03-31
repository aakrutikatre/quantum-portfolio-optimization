# quantum-portfolio-optimization
# Quantum Portfolio Optimization with Sentiment

This project uses quantum computing (Qiskit VQE) to optimize asset allocations, integrating sentiment scores from FinBERT and financial news (GDELT). It compares quantum optimization to classical optimizers using rolling window evaluations, Sharpe ratios, diversification, max drawdown, and Sortino ratios.

## 🧠 Techniques
- QUBO formulation with sentiment-adjusted returns
- Quantum optimization using SamplingVQE and custom TwoLocal ansatz
- Sentiment extraction via FinBERT on GDELT news headlines
- Weekly and monthly rolling rebalancing
- Evaluation metrics: Sharpe, Max Drawdown, Sortino, Runtime

## 📁 Main File

- `quantum_portfolio_main.py`: Complete end-to-end pipeline with all functions and evaluations.

## 🔧 Installation

```bash
pip install -r requirements.txt
