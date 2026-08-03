# What Drives the US Yield Curve?
### Level / slope / curvature factor analysis of 25 years of Fed policy and crises

![tests](https://github.com/nvb197/us-yield-curve-pca/actions/workflows/tests.yml/badge.svg)
![Python](https://img.shields.io/badge/Python-3.11%2B-3776AB?logo=python&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?logo=numpy&logoColor=white)
![pandas](https://img.shields.io/badge/pandas-150458?logo=pandas&logoColor=white)
![SQLite](https://img.shields.io/badge/SQLite-003B57?logo=sqlite&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?logo=scikitlearn&logoColor=white)

**Stack:** Python, NumPy, pandas, statsmodels, scikit-learn, SQLite, matplotlib, pytest, GitHub Actions

![The slope factor tracks the policy cycle](figures/00_hero_timeline.png)

Three factors explain about 95% of what the US Treasury curve does day to day.
I built the whole thing from scratch, including the eigendecomposition, then
used the factors to read back six Fed policy episodes.

I also started with a hypothesis that turned out to be wrong. I thought crises
squeeze the curve toward one dimension. They don't. They rotate it.

## Motivation
Litterman and Scheinkman found the three-factor structure in 1991. I rebuilt it
on data they never saw: daily par yield changes from 2000 to now, where they
used bond excess returns from the eighties. Same phenomenon, different data, so
this is a conceptual replication rather than a direct one.

Then two questions they couldn't ask in 1991. What do the factors say about
each Fed cycle since? And does the structure survive a crisis?

## The three factors
PC1 is a parallel level shift and takes 71.8% of the variance. PC2 is slope,
which is really the policy cycle, at 18.3%. PC3 is curvature at 5.1%. Loadings
are in `figures/02_loadings.png`.

Shape alone doesn't prove they are what I say they are, so I checked both
against quantities computed straight from the raw data. PC1 against the average
change across all eight maturities: 0.999. Cumulative PC2 against the 10Y–2Y
spread: 0.897.

## Reading 25 years of macro through the factors
For each episode I take the total curve move and project it onto the three
factors. Covariance basis, so the contributions come out in basis points.

Start and end dates come from documented events: FOMC decisions, the Bernanke
testimony, the unlimited-QE announcement. They are written into
`src/config.py::EPISODES` before I look at any output, not picked off the
chart. Residuals are reported for every episode, so if three factors don't
span a move, it shows.

| Episode | Regime | Factor signature |
|---|---|---|
| 2004–06 hikes | Bear flattening | PC1 +498 almost exactly cancelled by PC2 −498; long end barely moved |
| 2007–08 GFC | Bull steepening | PC1 −799, the largest level move in the sample; PC2 +451 |
| 2013 taper tantrum | Bear steepening | PC1 +162, PC2 +66, PC3 +45 — highest curvature share of the six |
| 2020 COVID | Level collapse | PC1 −285 in five weeks, with PC2 +175 alongside |
| 2022–23 inflation fight | Bear flattening | PC1 +688, PC2 −622; 537 straight days inverted |
| 2024– easing | Bear steepening | PC1 +184 with PC2 +136 — bear steepening, not the bull steepening I predicted |

## Does the structure hold still?
252-day rolling window, stepping a month at a time (`figures/04_rolling.png`).
Eigenvectors are only defined up to sign, and up to rotation when two
eigenvalues sit close together, so each window gets aligned against the
previous one.

I expected PC1 to spike above 90% in a crisis. That happens in cross-asset
risk and it's well documented. It does not happen here. PC1 stays between 70%
and 83% for the whole sample, and the highest reading is January 2002, not 2008
or 2020.

Measured in basis points the curve gets *more* multi-dimensional in a crisis,
not less. The dominant move is the Fed slamming the front end while the long
end sits still. That's a slope shock, and it adds variance orthogonal to level,
which pushes PC1's share down.

So the structure is more stable than I assumed. What actually breaks it is the
zero-rate era. From 2008 to 2017 PC1 explains only 63% and the curvature factor
sits 20.7 degrees off the full-sample one. Pin the front end at zero for years
and it stops moving, which is not something a correlation structure survives.

One caveat: consecutive windows share 231 of their 252 days, so this series is
heavily autocorrelated. It's descriptive, not a significance test.

## Appendix

Five checks behind the findings above. Each is one module, runnable on its own
with `python -m src.run_all --stages <name>`, and written up in
`notebooks/analysis.ipynb`.

- `hedging.py`: a duration-neutral 2s10s trade is immune to
  parallel shocks by construction but fully exposed to slope. A PC-hedge (two
  instruments, exposure orthogonal to PC1 and PC2) removes both, leaving only
  curvature risk. This is what the factor decomposition is for.
- `resampling.py`: a permutation test (shuffle each tenor's
  time order independently) puts the null EVR1 near 1/N ≈ 12.5%; the observed
  value lies entirely outside it. A block bootstrap gives a CI that respects
  autocorrelation.
- `sensitivity.py`: results re-run across tenor set, window
  length, subperiod, and correlation-vs-covariance basis. Where a claim doesn't
  survive, it's stated as a limitation.
- `nelson_siegel.py`: the Diebold-Li parametric model
  assumes the three shapes and fits them by OLS, sharing no machinery with PCA.
  Slope and curvature match at 0.83 and 0.79. Level only reaches 0.67, because
  NS forces a flat level loading while the empirical PC1 tilts toward longer
  maturities.
- `queries.py`: yearly summaries, longest inversion streaks (window
  functions), regime day-counts (LAG + CASE), computed inside SQLite.

## Data & design decisions
Eight constant-maturity tenors from FRED, 3M through 10Y, 2000 to now.

I left out the 1-month and the 30-year. The 1-month only starts in 2001 and is
noisy for reasons that have nothing to do with the term structure. The 30-year
has a worse problem: Treasury stopped issuing it from 2002 to 2006, so keeping
it would have quietly deleted four years from every other maturity too.

**Missing days are dropped, not forward-filled.** Filling them adds fake
zero-change days, which shrinks measured variance and distorts the correlation
structure, and a distorted correlation structure means distorted eigenvectors.
Dropping costs 4.1% of days.

I ran it both ways out of curiosity and the explained-variance numbers came out
identical. That makes sense: uniform zero rows just rescale the covariance
matrix, and scaling doesn't rotate eigenvectors. Free robustness check.

## Method
Work on daily changes, not levels. ADF says levels are I(1) and changes are
I(0), so differencing is both necessary and sufficient.

PCA runs through `numpy.linalg.eigh`, on the correlation matrix for the
structure view and on the covariance matrix where I need basis points (the
event study and the hedging section). I wrote the decomposition myself and then
checked it against scikit-learn. They agree to 1e-12.

The narrative with all the results is in `notebooks/analysis.ipynb`. Design
decisions and the traps behind them are in the docstrings. The 28 tests in
`tests/` cover PCA invariants, alignment, event decomposition, hedging
identities, resampling, robustness, and a set of degenerate inputs that should
fail loudly instead of quietly returning NaN.

## Limitations & outlook
PCA describes what happened. It has no dynamics and no no-arbitrage
constraint, so it can't price anything. Getting to pricing means affine term
structure models under the risk-neutral measure, Vasicek or CIR. That's where
I want to take this next.

The factors also drift. Up to 20.7 degrees in the zero-rate period, so
"constant structure" is only roughly true, and it's least true exactly when
things were most unusual.

One more on the data: DGS series are par yields, not zero-coupon. Fine for
looking at factor shapes, but any pricing work would need a fitted zero curve
such as the Fed's GSW dataset.

## Reproduce
```bash
pip install -r requirements.txt
export FRED_API_KEY=your_key        # free at fred.stlouisfed.org
python -m src.run_all               # fetch -> SQLite -> every figure and table
pytest -q                           # 28 tests, no network needed
```
`notebooks/analysis.ipynb` has the full narrative. It checks for
`FRED_API_KEY`: with a key it fetches real data, without one it falls back to
generated 3-factor data and labels every result as synthetic, so I can't quote
a fake number by mistake. To run one section at a time: `python -m src.run_all --stages
core,events,rolling` (also `hedging`, `stats`, `nelson_siegel`,
`sensitivity`, `sql`).

## Repository structure
```
.
├── src/
│   ├── config.py          # tenors, dates, the 6 macro episodes (anchored, sourced)
│   ├── etl.py             # FRED fetch, validation, drop-not-ffill cleaning, SQLite
│   ├── pca_engine.py      # hand-written eigendecomposition, sign convention
│   ├── rolling.py         # rolling PCA + eigenvector alignment
│   ├── events.py          # episode decomposition (ex-ante windows + residual)
│   ├── hedging.py         # duration-hedge vs PC-hedge P&L
│   ├── resampling.py      # permutation test + block bootstrap
│   ├── sensitivity.py     # robustness: tenor / window / subperiod / basis
│   ├── nelson_siegel.py   # Diebold-Li model, cross-checked against PCA
│   ├── queries.py         # analytical SQL layer
│   ├── plots.py           # every figure
│   ├── synthetic.py       # 3-factor synthetic data for tests and offline runs
│   └── run_all.py         # single entrypoint, 8 selectable stages
├── notebooks/
│   └── analysis.ipynb     # full narrative in English
├── tests/                 # 28 pytest cases, no network needed
├── figures/               # generated plots and tables
└── requirements.txt
```

## References
- Litterman, R. & Scheinkman, J. (1991). *Common Factors Affecting Bond Returns.* Journal of Fixed Income.
- Diebold, F.X. & Li, C. (2006). *Forecasting the Term Structure of Government Bond Yields.* Journal of Econometrics.
- Lord, R. & Pelsser, A. (2007). *Level–Slope–Curvature, Fact or Artefact?* Applied Mathematical Finance.