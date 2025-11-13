# Bilevel Index Tracking

This project implements a compact, finance-oriented demonstration of **sensitivity-based bilevel optimisation** for sparse index-tracking portfolios.

Goal: to show how a differentiable bilevel approach can select a small subset of assets to track a benchmark index (e.g. S&P 500) using a relaxed continuous selection vector and a convex lower-level tracking-error optimisation.

## Project Structure
blp-index-tracking/
├── prepare_data.py # Fetch price data from Stooq and build clean returns panel
├── data_io.py # Helper functions to load processed data
└── data/ # Output directory created by prepare_data.py
