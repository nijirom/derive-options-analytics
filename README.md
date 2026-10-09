# Derive Options Analytics

Interactive multi-currency options analytics built from the public [Derive v3 API](https://docs.derive.xyz/).

**Live dashboard:** https://nijirom.github.io/derive-options-analytics/

## Features

- 3D implied-volatility surfaces
- Volatility smiles and ATM term structure
- All active Derive option currencies
- Calls, puts, and OTM-composite views
- Strike-level fitted fair values
- Derive mark deviation from fitted fair value
- Live bid/ask deviations when quotes are available

## Fair-value model

For each currency, expiry, and option type, the generator:

1. Fits mark IV over log-moneyness with a leave-one-out local quadratic regression using neighboring strikes.
2. Prices each option with Black-76 using Derive's forward and discount factor.
3. Reports deviation as Derive mark minus fitted fair value.

This is an analytical estimate, not investment advice or an executable quote.

## Generate locally

Requires Python 3.11 or newer; no third-party packages or API credentials are needed.

```powershell
python derive_options_surface.py --currencies ALL --output index.html
```

Limit currencies or expiries when needed:

```powershell
python derive_options_surface.py --currencies BTC,SOL,XRP --max-expiries 6 --output index.html
```

The GitHub Pages workflow refreshes and deploys the dashboard every six hours.
