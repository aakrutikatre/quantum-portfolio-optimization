import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime
import time
import requests
from io import StringIO

from yahooquery import Ticker
from qiskit.circuit.library import TwoLocal
from qiskit_aer.primitives import Sampler
from qiskit_algorithms import SamplingVQE, QAOA
from qiskit_algorithms.optimizers import COBYLA
from qiskit_optimization import QuadraticProgram
from qiskit_optimization.algorithms import MinimumEigenOptimizer, CobylaOptimizer
from qiskit_finance.data_providers import RandomDataProvider
from transformers import pipeline

#########################################
#           CONFIGURATION               #
#########################################
# Data parameters
stocks = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "NVDA"]
start_date = "2018-01-01"  # Full historical period start
end_date = "2024-01-01"    # Full historical period end

# Define rebalancing frequencies and corresponding rolling window settings
frequency_settings = {
    "W": {"rolling_window": 52, "rolling_step": 13},  # Weekly: 52 weeks window, 13 weeks step
    "M": {"rolling_window": 12, "rolling_step": 3}      # Monthly: 12 months window, 3 months step
}

# Optimization parameter grid (for sensitivity analysis)
optimization_params = {
    "num_assets": 6,
    "bits_per_asset": 3,
    "precision": 2 ** 3,  # 2 ** bits_per_asset
    "delta": 1 / (2 ** 3),
    "budget": 1.0,
    "penalty": [5, 10, 20],
    "risk_factor": [0.2, 0.4, 0.6],
    "lambda_sentiment": [0.0, 0.5, 1.0]
}
# Set defaults (global variables for core optimization functions)
risk_factor = optimization_params["risk_factor"][1]       # default 0.4
penalty = optimization_params["penalty"][1]               # default 10
lambda_sentiment = optimization_params["lambda_sentiment"][1]  # default 0.5
num_assets = optimization_params["num_assets"]
bits_per_asset = optimization_params["bits_per_asset"]
precision = optimization_params["precision"]
delta = optimization_params["delta"]

# Local folder to save GDELT news data
news_data_path = "/Users/aakrutikatre/Documents/SEM4/News"
os.makedirs(news_data_path, exist_ok=True)

# Folder to save evaluation figures
figures_path = "/Users/aakrutikatre/Documents/SEM4/Evaluation_figures"
os.makedirs(figures_path, exist_ok=True)

#########################################
#         DATA FETCHING & PREPROCESSING #
#########################################
def fetch_price_data(frequency):
    print(f"Fetching historical price data at {frequency} frequency...")
    data = Ticker(stocks)
    df = data.history(start=start_date, end=end_date)["adjclose"]
    df = df.reset_index().pivot(index="date", columns="symbol", values="adjclose")
    df.index = pd.to_datetime(df.index)
    df = df.resample(frequency).last()  # Resample to chosen frequency
    returns = df.pct_change().dropna()
    print(f"Data loaded with {len(df)} records at {frequency} frequency.")
    return returns, df

# Fetch weekly and monthly full period data:
returns_weekly, df_weekly = fetch_price_data("W")
returns_monthly, df_monthly = fetch_price_data("M")

# For full period evaluation we define:
returns_full_weekly = returns_weekly.copy()
mu_full_weekly = np.array(returns_full_weekly.mean()).reshape(-1)
sigma_full_weekly = returns_full_weekly.cov().to_numpy()

returns_full_monthly = returns_monthly.copy()
mu_full_monthly = np.array(returns_full_monthly.mean()).reshape(-1)
sigma_full_monthly = returns_full_monthly.cov().to_numpy()

#########################################
#          GDELT NEWS RETRIEVAL         #
#########################################
def fetch_gdelt_news(ticker, start_date_str, end_date_str, max_records=100):
    filename = f"{ticker}_{start_date_str}_{end_date_str}.csv"
    filepath = os.path.join(news_data_path, filename)
    if os.path.exists(filepath):
        print(f"Loading cached news for {ticker} from {filepath}")
        try:
            df = pd.read_csv(filepath)
            return df
        except Exception as e:
            print(f"Error loading cached file {filepath}: {e}")
    url = "https://api.gdeltproject.org/api/v2/doc/doc"
    start_dt = start_date_str.replace("-", "") + "000000"
    end_dt = end_date_str.replace("-", "") + "235959"
    query = (f"{ticker} AND (earnings OR revenue OR profit OR growth OR decline OR forecast "
             f"OR downgrade OR upgrade OR 'financial performance' OR 'quarterly report' OR guidance "
             f"OR 'market share' OR valuation OR 'analyst rating' OR trading)")
    params = {
        'query': query,
        'mode': 'ArtList',
        'format': 'CSV',
        'maxrecords': max_records,
        'sort': 'DateDesc',
        'startdatetime': start_dt,
        'enddatetime': end_dt,
        'sourcelang': 'en',
        'sourcecountry': 'US'
    }
    print(f"Fetching GDELT news for {ticker} from {start_date_str} to {end_date_str}")
    response = requests.get(url, params=params)
    if response.status_code == 200:
        try:
            df = pd.read_csv(StringIO(response.text))
            df.to_csv(filepath, index=False)
            print(f"Saved news for {ticker} to {filepath}")
            return df
        except Exception as e:
            print(f"Error parsing CSV for {ticker}: {e}")
            return pd.DataFrame()
    else:
        print(f"Error fetching GDELT data for {ticker}: status code {response.status_code}")
        return pd.DataFrame()

def extract_headlines_from_gdelt(df):
    for col in ['DocumentTranslationHeadline', 'DocumentHeadline']:
        if col in df.columns:
            return df[col].dropna().tolist()
    return []

def compute_headline_weight(headline):
    keywords = ['earnings', 'revenue', 'profit', 'quarterly', 'guidance', 'forecast', 'downgrade', 'upgrade', 'growth']
    weight = 1.0
    headline_lower = headline.lower()
    for kw in keywords:
        if kw in headline_lower:
            weight += 0.5
    return weight

def fetch_window_sentiment(window_start, window_end, stocks, max_records=20):
    sentiment_scores = []
    from_date = window_start.strftime("%Y-%m-%d")
    to_date = window_end.strftime("%Y-%m-%d")
    for ticker in stocks:
        df_news = fetch_gdelt_news(ticker, from_date, to_date, max_records=max_records)
        headlines = extract_headlines_from_gdelt(df_news)
        combined_score = 0
        total_weight = 0
        for headline in headlines:
            result = sentiment_model(headline)[0]
            score = result["score"] if result["label"].lower() == "positive" else -result["score"]
            weight = compute_headline_weight(headline)
            combined_score += score * weight
            total_weight += weight
        avg_score = combined_score / total_weight if total_weight > 0 else 0.0
        sentiment_scores.append(avg_score)
    return np.array(sentiment_scores)

# For full-period sentiment using GDELT with richer signals
full_sentiment_scores = []
full_from_date = pd.to_datetime(start_date).strftime("%Y-%m-%d")
full_to_date = pd.to_datetime(end_date).strftime("%Y-%m-%d")
sentiment_model = pipeline("sentiment-analysis", model="ProsusAI/finbert")
for ticker in stocks:
    df_news = fetch_gdelt_news(ticker, full_from_date, full_to_date, max_records=50)
    headlines = extract_headlines_from_gdelt(df_news)
    combined_score = 0
    total_weight = 0
    for headline in headlines:
        result = sentiment_model(headline)[0]
        score = result["score"] if result["label"].lower() == "positive" else -result["score"]
        weight = compute_headline_weight(headline)
        combined_score += score * weight
        total_weight += weight
    full_sentiment_scores.append(combined_score / total_weight if total_weight > 0 else 0.0)
full_sentiment_scores = np.array(full_sentiment_scores)
mu_sent_full = mu_full_weekly + lambda_sentiment * full_sentiment_scores

#########################################
#          FUNCTION DEFINITIONS         #
#########################################
def build_qubo(mu_vector, sigma):
    qp = QuadraticProgram(name="QuantumPortfolio")
    var_names = [f"x_{i}_{j}" for i in range(num_assets) for j in range(bits_per_asset)]
    for name in var_names:
        qp.binary_var(name)
    weight_exprs = {
        i: [(f"x_{i}_{j}", delta * (2 ** (bits_per_asset - j - 1))) for j in range(bits_per_asset)]
        for i in range(num_assets)
    }
    quadratic_terms = {}
    linear_terms = {}
    for i in range(num_assets):
        for j in range(num_assets):
            cov = sigma[i, j]
            if abs(cov) < 1e-8:
                continue
            for bi, wi in weight_exprs[i]:
                for bj, wj in weight_exprs[j]:
                    key = (bi, bj)
                    quadratic_terms[key] = quadratic_terms.get(key, 0) + risk_factor * wi * wj * cov
    for i in range(num_assets):
        for bi, wi in weight_exprs[i]:
            linear_terms[bi] = linear_terms.get(bi, 0) - (1 - risk_factor) * wi * mu_vector[i]
    all_weights = [(bi, wi) for expr in weight_exprs.values() for (bi, wi) in expr]
    for bi, wi in all_weights:
        for bj, wj in all_weights:
            key = (bi, bj)
            quadratic_terms[key] = quadratic_terms.get(key, 0) + penalty * wi * wj
    for bi, wi in all_weights:
        linear_terms[bi] = linear_terms.get(bi, 0) - 2 * penalty * wi
    qp.minimize(linear=linear_terms, quadratic=quadratic_terms, constant=penalty * 1.0)
    return qp, var_names

def solve_quantum(qp, var_names, custom_ansatz=None):
    losses = []
    def callback(eval_count, params, mean, stddev):
        losses.append(mean)
    ansatz = custom_ansatz if custom_ansatz is not None else TwoLocal(len(var_names), "ry", "cz", reps=2, entanglement="full")
    cobyla = COBYLA()
    cobyla.set_options(maxiter=200)
    svqe_mes = SamplingVQE(sampler=Sampler(), ansatz=ansatz, optimizer=cobyla, callback=callback)
    solver = MinimumEigenOptimizer(svqe_mes)
    start = time.time()
    result = solver.solve(qp)
    runtime = time.time() - start
    bitstring = np.array([result.variables_dict[name] for name in var_names])
    weights = np.zeros(num_assets)
    for i in range(num_assets):
        for j in range(bits_per_asset):
            weights[i] += bitstring[i * bits_per_asset + j] * delta * (2 ** (bits_per_asset - j - 1))
    return weights, losses, runtime

def solve_classical(mu_vector, sigma):
    qp = QuadraticProgram(name="ClassicalPortfolio")
    for i in range(num_assets):
        qp.continuous_var(name=f"w{i}", lowerbound=0.0, upperbound=1.0)
    qp.linear_constraint({f"w{i}": 1.0 for i in range(num_assets)}, sense="==", rhs=1.0, name="budget")
    for i in range(num_assets):
        qp.linear_constraint({f"w{i}": 1.0}, sense="<=", rhs=0.5, name=f"max_w{i}")
    linear = {f"w{i}": - (1 - risk_factor) * mu_vector[i] for i in range(num_assets)}
    quadratic = {(f"w{i}", f"w{j}"): risk_factor * sigma[i, j]
                 for i in range(num_assets) for j in range(num_assets)
                 if abs(sigma[i, j]) > 1e-8}
    qp.minimize(linear=linear, quadratic=quadratic)
    start = time.time()
    result = CobylaOptimizer().solve(qp)
    runtime = time.time() - start
    weights = np.array([result.variables_dict[f"w{i}"] for i in range(num_assets)])
    return weights, runtime

def compute_kpis(weights, mu_vector, sigma, label=""):
    weights_clean = np.where(np.abs(weights) < 1e-8, 0, weights)
    expected_return = (weights_clean @ mu_vector) * 100
    risk_val = (weights_clean @ sigma @ weights_clean) * 100
    sharpe = expected_return / np.sqrt(risk_val) if risk_val > 0 else 0
    diversification = np.sum(weights_clean > 1e-4)
    print(f"\n📊 {label} Portfolio KPIs")
    print(f"Weights: {np.round(weights_clean, 4)}")
    print(f"Expected Return: {expected_return:.2f}%")
    print(f"Portfolio Risk:  {risk_val:.2f}%")
    print(f"Sharpe Ratio:    {sharpe:.4f}")
    print(f"Diversification: {int(diversification)} assets")
    return expected_return, risk_val, sharpe

def compute_max_drawdown(portfolio_values):
    running_max = np.maximum.accumulate(portfolio_values)
    drawdowns = (portfolio_values - running_max) / running_max
    return drawdowns.min()

def simulate_portfolio_performance(weights, returns_window):
    portfolio_returns = returns_window.dot(weights)
    portfolio_values = np.cumprod(1 + portfolio_returns)
    return portfolio_values

def compute_sortino_ratio(weights, returns_window, risk_free=0.0):
    portfolio_returns = returns_window.dot(weights)
    downside_returns = portfolio_returns[portfolio_returns < risk_free]
    downside_deviation = np.std(downside_returns) if len(downside_returns) > 0 else 0.0
    avg_return = np.mean(portfolio_returns)
    return (avg_return - risk_free) / downside_deviation if downside_deviation > 0 else np.nan

def compute_additional_risk_measures(weights, returns_window, risk_free=0.0):
    portfolio_values = simulate_portfolio_performance(weights, returns_window)
    max_dd = compute_max_drawdown(portfolio_values)
    sortino = compute_sortino_ratio(weights, returns_window, risk_free)
    return max_dd, sortino

#########################################
#         ADVANCED QUANTUM MODULE       #
#########################################
quantum_configs = [
    {'reps': 1, 'entanglement': 'linear'},
    {'reps': 2, 'entanglement': 'linear'},
    {'reps': 2, 'entanglement': 'full'},
    {'reps': 3, 'entanglement': 'full'}
]

def evaluate_quantum_configs(mu_vector, sigma):
    results = []
    convergence_data = {}
    for config in quantum_configs:
        print(f"\nEvaluating quantum config: reps={config['reps']}, entanglement={config['entanglement']}")
        qp, var_names = build_qubo(mu_vector, sigma)
        ansatz = TwoLocal(len(var_names), "ry", "cz", reps=config['reps'], entanglement=config['entanglement'])
        weights, losses, runtime = solve_quantum(qp, var_names, custom_ansatz=ansatz)
        exp_return, risk_val, sharpe = compute_kpis(weights, mu_vector, sigma, label=f"Quantum Config {config}")
        results.append({
            'config': config,
            'expected_return': exp_return,
            'risk': risk_val,
            'sharpe': sharpe,
            'runtime': runtime,
            'final_loss': losses[-1] if losses else None,
            'weights': weights
        })
        convergence_data[f"reps={config['reps']}, entanglement={config['entanglement']}"] = losses
    return results, convergence_data

#########################################
#      ROLLING & WEIGHT EVOLUTION       #
#########################################
def run_evaluation(returns, frequency, use_sentiment=False, use_quantum=True):
    settings = frequency_settings[frequency]
    rolling_window = settings["rolling_window"]
    rolling_step = settings["rolling_step"]

    dates = []
    quantum_results = []
    classical_results = []
    periods = returns.index
    start_idx = 0
    max_records = 50 if frequency == "M" else 20

    while start_idx + rolling_window <= len(periods):
        window_dates = periods[start_idx: start_idx + rolling_window]
        window_data = returns.loc[window_dates]
        window_end = window_data.index[-1]
        dates.append(window_end)
        mu_window = np.array(window_data.mean()).reshape(-1)
        sigma_window = window_data.cov().to_numpy()
        if use_sentiment:
            window_start = window_dates[0]
            window_sentiment = fetch_window_sentiment(window_start, window_end, stocks, max_records=max_records)
            mu_window = mu_window + lambda_sentiment * window_sentiment
        # Quantum Optimization
        if use_quantum:
            qp_q, var_names = build_qubo(mu_window, sigma_window)
            start_time_q = time.time()
            weights_q, losses, _ = solve_quantum(qp_q, var_names)
            runtime_q = time.time() - start_time_q
            weights_q_clean = np.where(np.abs(weights_q) < 1e-8, 0, weights_q)
            print(f"Quantum weights for window ending {window_end}: {np.round(weights_q_clean, 4)}")
            exp_return_q, risk_q, sharpe_q = compute_kpis(weights_q, mu_window, sigma_window, label="Quantum Rolling")
            max_dd_q, sortino_q = compute_additional_risk_measures(weights_q, window_data)
            diversification_q = np.sum(weights_q > 1e-4)
            quantum_results.append({
                'date': window_end,
                'expected_return': exp_return_q,
                'risk': risk_q,
                'sharpe': sharpe_q,
                'diversification': diversification_q,
                'runtime': runtime_q,
                'final_loss': losses[-1] if losses else None,
                'max_drawdown': max_dd_q,
                'sortino': sortino_q
            })
        else:
            quantum_results.append({
                'date': window_end,
                'expected_return': None,
                'risk': None,
                'sharpe': None,
                'diversification': None,
                'runtime': None,
                'final_loss': None,
                'max_drawdown': None,
                'sortino': None
            })
        # Classical Optimization
        start_time_c = time.time()
        weights_c, _ = solve_classical(mu_window, sigma_window)
        runtime_c = time.time() - start_time_c
        weights_c_clean = np.where(np.abs(weights_c) < 1e-8, 0, weights_c)
        print(f"Classical weights for window ending {window_end}: {np.round(weights_c_clean, 4)}")
        exp_return_c, risk_c, sharpe_c = compute_kpis(weights_c, mu_window, sigma_window, label="Classical Rolling")
        max_dd_c, sortino_c = compute_additional_risk_measures(weights_c, window_data)
        diversification_c = np.sum(weights_c > 1e-4)
        classical_results.append({
            'date': window_end,
            'expected_return': exp_return_c,
            'risk': risk_c,
            'sharpe': sharpe_c,
            'diversification': diversification_c,
            'runtime': runtime_c,
            'max_drawdown': max_dd_c,
            'sortino': sortino_c
        })
        start_idx += rolling_step
    df_quantum = pd.DataFrame(quantum_results)
    df_classical = pd.DataFrame(classical_results)
    return df_quantum, df_classical

def run_evaluation_with_weights(returns, frequency, use_sentiment=False, use_quantum=True):
    settings = frequency_settings[frequency]
    rolling_window = settings["rolling_window"]
    rolling_step = settings["rolling_step"]

    dates = []
    quantum_weights_list = []
    classical_weights_list = []
    periods = returns.index
    start_idx = 0
    max_records = 50 if frequency == "M" else 20

    while start_idx + rolling_window <= len(periods):
        window_dates = periods[start_idx: start_idx + rolling_window]
        window_data = returns.loc[window_dates]
        window_end = window_data.index[-1]
        dates.append(window_end)
        mu_window = np.array(window_data.mean()).reshape(-1)
        sigma_window = window_data.cov().to_numpy()
        if use_sentiment:
            window_start = window_dates[0]
            window_sentiment = fetch_window_sentiment(window_start, window_end, stocks, max_records=max_records)
            mu_window = mu_window + lambda_sentiment * window_sentiment
        if use_quantum:
            qp_q, var_names = build_qubo(mu_window, sigma_window)
            weights_q, _, _ = solve_quantum(qp_q, var_names)
            quantum_weights_list.append(weights_q)
        else:
            quantum_weights_list.append(None)
        weights_c, _ = solve_classical(mu_window, sigma_window)
        classical_weights_list.append(weights_c)
        start_idx += rolling_step
    return dates, quantum_weights_list, classical_weights_list

#########################################
#             SENSITIVITY ANALYSIS      #
#########################################
def sensitivity_analysis_all(mu_vector, sigma):
    risk_factors = optimization_params["risk_factor"]
    penalties = optimization_params["penalty"]
    sentiment_weights = optimization_params["lambda_sentiment"]
    results = []
    for rf in risk_factors:
        for pen in penalties:
            for lam in sentiment_weights:
                mu_adjusted = mu_vector + lam * full_sentiment_scores
                global risk_factor, penalty, lambda_sentiment
                orig_rf, orig_pen, orig_lam = risk_factor, penalty, lambda_sentiment
                risk_factor, penalty, lambda_sentiment = rf, pen, lam
                weights_c, _ = solve_classical(mu_adjusted, sigma)
                exp_ret_c, risk_c, sharpe_c = compute_kpis(weights_c, mu_adjusted, sigma,
                                                             label=f"Classical: rf={rf}, pen={pen}, lam={lam}")
                qp_q, var_names = build_qubo(mu_adjusted, sigma)
                weights_q, _, _ = solve_quantum(qp_q, var_names)
                exp_ret_q, risk_q, sharpe_q = compute_kpis(weights_q, mu_adjusted, sigma,
                                                             label=f"Quantum: rf={rf}, pen={pen}, lam={lam}")
                results.append({
                    'method': 'Classical',
                    'risk_factor': rf,
                    'penalty': pen,
                    'lambda_sentiment': lam,
                    'expected_return': exp_ret_c,
                    'risk': risk_c,
                    'sharpe': sharpe_c
                })
                results.append({
                    'method': 'Quantum',
                    'risk_factor': rf,
                    'penalty': pen,
                    'lambda_sentiment': lam,
                    'expected_return': exp_ret_q,
                    'risk': risk_q,
                    'sharpe': sharpe_q
                })
                risk_factor, penalty, lambda_sentiment = orig_rf, orig_pen, orig_lam
    return pd.DataFrame(results)

#########################################
#             MAIN EXECUTION            #
#########################################
if __name__ == "__main__":
    # 1. Full Period Evaluation (Base and with Sentiment) using weekly frequency
    print("\n=== Full Period Evaluation (Base, Weekly) ===")
    qp_base, vars_base = build_qubo(mu_full_weekly, sigma_full_weekly)
    weights_q_base, losses_base, time_q_base = solve_quantum(qp_base, vars_base)
    weights_c_base, time_c_base = solve_classical(mu_full_weekly, sigma_full_weekly)
    weights_q_base_clean = np.where(np.abs(weights_q_base) < 1e-8, 0, weights_q_base)
    weights_c_base_clean = np.where(np.abs(weights_c_base) < 1e-8, 0, weights_c_base)
    print("Quantum weights (Full, Weekly):", np.round(weights_q_base_clean, 4))
    print("Classical weights (Full, Weekly):", np.round(weights_c_base_clean, 4))
    compute_kpis(weights_q_base, mu_full_weekly, sigma_full_weekly, label="Quantum (Full, Weekly)")
    compute_kpis(weights_c_base, mu_full_weekly, sigma_full_weekly, label="Classical (Full, Weekly)")

    print("\n=== Full Period Evaluation (With Sentiment, Weekly) ===")
    qp_sent, vars_sent = build_qubo(mu_sent_full, sigma_full_weekly)
    weights_q_sent, losses_sent, time_q_sent = solve_quantum(qp_sent, vars_sent)
    weights_c_sent, time_c_sent = solve_classical(mu_sent_full, sigma_full_weekly)
    weights_q_sent_clean = np.where(np.abs(weights_q_sent) < 1e-8, 0, weights_q_sent)
    weights_c_sent_clean = np.where(np.abs(weights_c_sent) < 1e-8, 0, weights_c_sent)
    print("Quantum weights (Sentiment Full, Weekly):", np.round(weights_q_sent_clean, 4))
    print("Classical weights (Sentiment Full, Weekly):", np.round(weights_c_sent_clean, 4))
    compute_kpis(weights_q_sent, mu_sent_full, sigma_full_weekly, label="Quantum (Sentiment Full, Weekly)")
    compute_kpis(weights_c_sent, mu_sent_full, sigma_full_weekly, label="Classical (Sentiment Full, Weekly)")

    # 1b. Full Period Evaluation for Monthly Data
    print("\n=== Full Period Evaluation (Base, Monthly) ===")
    qp_base_m, vars_base_m = build_qubo(mu_full_monthly, sigma_full_monthly)
    weights_q_base_m, losses_base_m, time_q_base_m = solve_quantum(qp_base_m, vars_base_m)
    weights_c_base_m, time_c_base_m = solve_classical(mu_full_monthly, sigma_full_monthly)
    weights_q_base_m_clean = np.where(np.abs(weights_q_base_m) < 1e-8, 0, weights_q_base_m)
    weights_c_base_m_clean = np.where(np.abs(weights_c_base_m) < 1e-8, 0, weights_c_base_m)
    print("Quantum weights (Full, Monthly):", np.round(weights_q_base_m_clean, 4))
    print("Classical weights (Full, Monthly):", np.round(weights_c_base_m_clean, 4))
    compute_kpis(weights_q_base_m, mu_full_monthly, sigma_full_monthly, label="Quantum (Full, Monthly)")
    compute_kpis(weights_c_base_m, mu_full_monthly, sigma_full_monthly, label="Classical (Full, Monthly)")

    # 2. Rolling Analysis Evaluation for both Weekly and Monthly frequencies
    for freq in frequency_settings.keys():
        print(f"\n=== Rolling Analysis Evaluation ({freq}) with Sentiment and Additional Risk Measures ===")
        returns_freq, _ = fetch_price_data(freq)
        max_records = 50 if freq == "M" else 20
        df_quantum, df_classical = run_evaluation(returns_freq, freq, use_sentiment=True, use_quantum=True)
        print(f"\nQuantum Rolling Evaluation ({freq}) (first 5 rows):")
        print(df_quantum.head())
        print(f"\nClassical Rolling Evaluation ({freq}) (first 5 rows):")
        print(df_classical.head())

        # Plot Rolling Sharpe Ratio and save figure
        plt.figure(figsize=(10, 5))
        plt.plot(df_quantum['date'], df_quantum['sharpe'], marker='o', label='Quantum')
        plt.plot(df_classical['date'], df_classical['sharpe'], marker='o', label='Classical')
        plt.xlabel('Window End Date')
        plt.ylabel('Sharpe Ratio')
        plt.title(f'Rolling Sharpe Ratio Evaluation ({freq})')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(figures_path, f'rolling_sharpe_{freq}.png'), dpi=300)
        plt.show()

        # Plot Maximum Drawdown and Sortino Ratio and save figure
        fig, axs = plt.subplots(2, 1, figsize=(10, 8))
        axs[0].plot(df_quantum['date'], df_quantum['max_drawdown'], marker='o', label='Quantum')
        axs[0].plot(df_classical['date'], df_classical['max_drawdown'], marker='o', label='Classical')
        axs[0].set_title(f'Rolling Maximum Drawdown ({freq})')
        axs[0].set_ylabel('Max Drawdown')
        axs[0].legend()
        axs[0].grid(True)

        axs[1].plot(df_quantum['date'], df_quantum['sortino'], marker='o', label='Quantum')
        axs[1].plot(df_classical['date'], df_classical['sortino'], marker='o', label='Classical')
        axs[1].set_title(f'Rolling Sortino Ratio ({freq})')
        axs[1].set_xlabel('Window End Date')
        axs[1].set_ylabel('Sortino Ratio')
        axs[1].legend()
        axs[1].grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(figures_path, f'rolling_drawdown_sortino_{freq}.png'), dpi=300)
        plt.show()

        df_quantum.to_csv(f"/Users/aakrutikatre/Documents/SEM4/Evaluation_csvs/quantum_rolling_evaluation_{freq}.csv", index=False)
        df_classical.to_csv(f"/Users/aakrutikatre/Documents/SEM4/Evaluation_csvs/classical_rolling_evaluation_{freq}.csv", index=False)

    # 3. Sensitivity Analysis for both Classical and Quantum (Weekly)
    print("\n=== Sensitivity Analysis (Weekly) ===")
    df_sensitivity = sensitivity_analysis_all(mu_full_weekly, sigma_full_weekly)
    print(df_sensitivity)
    df_sensitivity.to_csv("/Users/aakrutikatre/Documents/SEM4/Evaluation_csvs/sensitivity_analysis.csv", index=False)
    # Save sensitivity heatmap/line plot (example: plot Sharpe ratio vs. sentiment weight for rf=0.2, pen=5)
    sensitivity_subset = df_sensitivity[(df_sensitivity['risk_factor'] == 0.2) & (df_sensitivity['penalty'] == 5)]
    plt.figure(figsize=(8, 5))
    for method in sensitivity_subset['method'].unique():
        subset = sensitivity_subset[sensitivity_subset['method'] == method]
        plt.plot(subset['lambda_sentiment'], subset['sharpe'], marker='o', label=method)
    plt.xlabel('Sentiment Weight (λ)')
    plt.ylabel('Sharpe Ratio')
    plt.title('Sensitivity Analysis (rf=0.2, pen=5)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(figures_path, 'sensitivity_rf0.2_pen5.png'), dpi=300)
    plt.show()

    # 4. Maximum Drawdown Example for Full Period (using classical weights, weekly)
    print("\n=== Maximum Drawdown Example (Full Period, Weekly) ===")
    window_dates = returns_full_weekly.index[-frequency_settings["W"]["rolling_window"]:]
    returns_window = returns_full_weekly.loc[window_dates]
    portfolio_values = simulate_portfolio_performance(weights_c_base, returns_window)
    max_dd = compute_max_drawdown(portfolio_values)
    print(f"Maximum Drawdown (Full Period Classical, Weekly): {max_dd:.4f}")

    # 5. Tracking Portfolio Weight Evolution for Weekly frequency
    print("\n=== Tracking Portfolio Weight Evolution (Weekly) ===")
    dates_weights, quantum_weights, classical_weights = run_evaluation_with_weights(returns_full_weekly, "W", use_sentiment=True, use_quantum=True)
    classical_weights_array = np.array(classical_weights)
    plt.figure(figsize=(10, 5))
    for i in range(num_assets):
        plt.plot(dates_weights, classical_weights_array[:, i], marker='o', label=f"{stocks[i]}")
    plt.xlabel("Window End Date")
    plt.ylabel("Asset Weight")
    plt.title("Evolution of Classical Portfolio Weights (Weekly)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(figures_path, 'classical_weights_evolution_weekly.png'), dpi=300)
    plt.show()

    # 6. Advanced Quantum: Evaluate Different Circuit Architectures (using full period weekly data)
    print("\n=== Advanced Quantum Circuit Architectures Evaluation (Weekly) ===")
    quantum_config_results, convergence_data = evaluate_quantum_configs(mu_full_weekly, sigma_full_weekly)
    for res in quantum_config_results:
        print(f"Config {res['config']} -> Sharpe: {res['sharpe']:.4f}, Runtime: {res['runtime']:.2f}s, Final Loss: {res['final_loss']}")
    plt.figure(figsize=(10, 6))
    for config_label, losses in convergence_data.items():
        if losses is not None and len(losses) > 0:
            plt.plot(losses, marker='o', label=config_label)
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.title("Quantum Convergence Behavior for Different Configurations (Weekly)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(figures_path, 'quantum_convergence_weekly.png'), dpi=300)
    plt.show()

    # 7. (Optional) Full Period Evaluation: Save bar charts for asset weight distributions.
    # Weekly Full-Period Weights Bar Chart
    fig, ax = plt.subplots(figsize=(8, 5))
    ind = np.arange(len(stocks))
    width = 0.35
    ax.bar(ind, weights_c_base_clean, width, label='Classical')
    ax.bar(ind + width, weights_q_base_clean, width, label='Quantum')
    ax.set_ylabel('Weight')
    ax.set_title('Full Period (Weekly) Asset Weight Distribution')
    ax.set_xticks(ind + width / 2)
    ax.set_xticklabels(stocks)
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(figures_path, 'full_period_weekly_weights.png'), dpi=300)
    plt.show()

    # Monthly Full-Period Weights Bar Chart
    fig, ax = plt.subplots(figsize=(8, 5))
    # For monthly, recompute weights using monthly full period data:
    qp_base_m, vars_base_m = build_qubo(mu_full_monthly, sigma_full_monthly)
    weights_q_base_m, _, _ = solve_quantum(qp_base_m, vars_base_m)
    weights_c_base_m, _ = solve_classical(mu_full_monthly, sigma_full_monthly)
    weights_q_base_m_clean = np.where(np.abs(weights_q_base_m) < 1e-8, 0, weights_q_base_m)
    weights_c_base_m_clean = np.where(np.abs(weights_c_base_m) < 1e-8, 0, weights_c_base_m)
    ax.bar(ind, weights_c_base_m_clean, width, label='Classical')
    ax.bar(ind + width, weights_q_base_m_clean, width, label='Quantum')
    ax.set_ylabel('Weight')
    ax.set_title('Full Period (Monthly) Asset Weight Distribution')
    ax.set_xticks(ind + width / 2)
    ax.set_xticklabels(stocks)
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(figures_path, 'full_period_monthly_weights.png'), dpi=300)
    plt.show()

    print("\nAll evaluations completed and results saved.")
