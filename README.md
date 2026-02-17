# VIX-Style Volatility Index Calculator

A Python implementation aligned with the CBOE VIX methodology, adapted for commodity options using settlement prices as a proxy for mid quotes.

## Overview

This project computes a VIX-style implied volatility index from an options chain, following the methodology described in the [CBOE VIX White Paper](http://www.cboe.com/micro/vix/vixwhite.pdf).

**Key Features:**
- CBOE-aligned variance swap replication formula
- Two expirations bracketing the target maturity (30 days by default)
- Support for different underlying assets
- Clean, tested, desk-ready code

## Mathematical Methodology

### 1. Forward Price (F) and ATM Strike (K₀)

For each expiration term:

1. Find strike K* that minimizes |Call(K*) - Put(K*)|
2. Compute forward: `F = K* + exp(rT) × (Call(K*) - Put(K*))`
3. ATM strike: `K₀ = max{K : K ≤ F}`

### 2. Option Selection Q(K)

Following CBOE methodology, we construct Q(K) using out-of-the-money options:

- **K < K₀**: Use put prices
- **K > K₀**: Use call prices  
- **K = K₀**: Use average: Q(K₀) = (Call(K₀) + Put(K₀)) / 2

**Exclusion rule:** Stop including strikes after 2 consecutive zero-price quotes (from K₀ outward).

### 3. Strike Intervals (ΔK)

```
ΔKᵢ = (Kᵢ₊₁ - Kᵢ₋₁) / 2    for interior strikes
ΔK₁ = K₂ - K₁              for first strike
ΔKₙ = Kₙ - Kₙ₋₁            for last strike
```

### 4. Variance Contribution (σ²)

The variance for each term is computed as:

```
σ² = (2/T) × Σᵢ (ΔKᵢ/Kᵢ²) × e^(rT) × Q(Kᵢ) - (1/T) × (F/K₀ - 1)²
```

### 5. VIX Interpolation

The final VIX interpolates between near-term (T₁) and next-term (T₂) variances to a target maturity:

```
VIX = 100 × √[ (w₁ + w₂) × (N_year / N_target) ]

where:
  N₁, N₂, N_target, N_year = time in minutes
  Tᵢ = Nᵢ / N_year (time in years)
  w₁ = T₁ × σ₁² × (N₂ - N_target) / (N₂ - N₁)
  w₂ = T₂ × σ₂² × (N_target - N₁) / (N₂ - N₁)
```

## Key Assumptions & Differences from Official VIX

| Aspect | Official VIX (SPX) | This Implementation |
|:-------|:-------------------|:--------------------|
| Price source | Live bid-ask midpoint | Settlement prices (priorSettle) |
| Underlying | S&P 500 options | Commodity options (configurable) |
| Zero-bid rule | 2 consecutive zero bids | 2 consecutive zero settles |
| Time horizon | Fixed 30-day | Configurable |

**Why priorSettle?** Without access to live bid-ask quotes, we use the previous day's settlement price as a proxy for the option's fair value. This is a common approximation in academic and backtesting contexts.

## Project Structure

```
vix-calculator/
├── src/
│   ├── __init__.py
│   └── vix_index.py      # Core calculation module
├── tests/
│   ├── __init__.py
│   └── test_vix_index.py # Unit tests
├── data/                  # Place CSV files here
│   └── .gitkeep
├── run.py                 # Entry point
├── requirements.txt
├── .gitignore
└── README.md
```

## Installation

```bash
# Clone the repository
git clone <repository-url>
cd vix-calculator

# Create virtual environment (recommended)
python -m venv venv

# Activate (Windows PowerShell)
.\venv\Scripts\Activate.ps1

# Activate (Windows cmd)
venv\Scripts\activate.bat

# Activate (Unix/macOS)
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

## Required Data Files

Place the following CSV files in the `data/` directory:

1. **henry hub european options.csv** - Options chain with columns:
   - `options-id`: Product identifier
   - `tradeDate`: Trading date
   - `futures-expirationDate`: Expiration month
   - `options-strikePrice`: Strike price
   - `options-optiontype`: "call" or "put"
   - `options-priorSettle`: Settlement price
   - `futures-updated`: Timestamp

2. **treasury yield curve rates.csv** - CMT rates with columns:
   - `Date`: Date
   - `maturity`: e.g., "1 Mo", "2 Mo", "3 Mo"
   - `value`: Rate in percentage points

3. **cme holidays.csv** - Holiday calendar with column:
   - `DATE`: Holiday dates

## Usage

### Basic Execution

```bash
python run.py
```

### Example Output

Example output (illustrative; actual values depend on dataset and date):

```
============================================================
VIX-STYLE VOLATILITY INDEX CALCULATION
============================================================

VIX Index:  42.15

------------------------------------------------------------
NEAR-TERM CONTRACT
------------------------------------------------------------
  Time to expiration (T1):  0.054795 years
  Forward price (F1):       2.8750
  ATM strike (K0_1):        2.8500
  Variance (σ²_1):          0.175432
  Risk-free rate (r1):      0.0800%
  Strikes used:             23

------------------------------------------------------------
NEXT-TERM CONTRACT
------------------------------------------------------------
  Time to expiration (T2):  0.136986 years
  Forward price (F2):       2.9125
  ATM strike (K0_2):        2.9000
  Variance (σ²_2):          0.182156
  Risk-free rate (r2):      0.0900%
  Strikes used:             25

============================================================
```

### Programmatic Usage

```python
import pandas as pd
from src.vix_index import vix_index

# Load your data
options = pd.read_csv("data/henry hub european options.csv")
rates = pd.read_csv("data/treasury yield curve rates.csv")
calendar = pd.read_csv("data/cme holidays.csv")

# Convert dates
options["futures-expirationDate"] = pd.to_datetime(options["futures-expirationDate"])
options["tradeDate"] = pd.to_datetime(options["tradeDate"])
options["futures-updated"] = pd.to_datetime(options["futures-updated"])
rates["Date"] = pd.to_datetime(rates["Date"])

# Calculate VIX (90-day horizon)
result = vix_index(
    options_df=options,
    rates_df=rates,
    calendar_df=calendar,
    options_id=1352,
    trade_date="2020-11-12",
    front_months=2,
    rear_months=4,
    target_days=90,
    price_scale=100,  # dataset-dependent: aligns strike/price units
)

print(f"VIX: {result['vix']:.2f}")
```

## Running Tests

```bash
pytest tests/ -v
```

Expected output:
```
tests/test_vix_index.py::TestComputeDeltaK::test_interior_strikes PASSED
tests/test_vix_index.py::TestComputeDeltaK::test_boundary_strikes PASSED
tests/test_vix_index.py::TestComputeForwardAndK0::test_forward_greater_than_k0 PASSED
tests/test_vix_index.py::TestBuildQK::test_k0_included_once PASSED
...
```

## Configuration Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `options_id` | Product identifier in options data | - |
| `trade_date` | Calculation date | - |
| `front_months` | Near-term expiration (months ahead from trade_date) | 1 |
| `rear_months` | Next-term expiration (months ahead from trade_date) | 2 |
| `expiration_hour` | Settlement hour (0-23) | 16 |
| `expiration_day_offset` | Business day from month-end | 4 |
| `target_days` | Target maturity for interpolation | 30 |
| `price_scale` | Scale factor to align strike and option price units (dataset-dependent) | 1.0 |

## References

- [CBOE VIX White Paper](http://www.cboe.com/micro/vix/vixwhite.pdf)
- [The Fear Index: VIX and Variance Swaps](https://berentlunde.netlify.app/post/the-fear-index-vix-and-variance-swaps)
- [More Than You Ever Wanted to Know About Volatility Swaps](https://www.researchgate.net/publication/246869706_More_Than_You_Ever_Wanted_to_Know_About_Volatility_Swaps)

## License

MIT License
