import numpy as np

def _positive_probability(model, X):
    return model.predict_proba(X)[:, 1]

def compute_ice(model, X_test, feature, n_points=50, n_samples=60, random_state=42):
    values = X_test[feature].to_numpy()
    low, high = np.quantile(values, [0.02, 0.98])
    grid = np.linspace(low, high, n_points)

    rng = np.random.default_rng(random_state)
    n = min(n_samples, len(X_test))
    chosen = rng.choice(len(X_test), size=n, replace=False)
    X_base = X_test.iloc[chosen].copy()

    curves = np.zeros((n, n_points), dtype=float)

    for j, value in enumerate(grid):
        X_modified = X_base.copy()
        X_modified[feature] = value
        curves[:, j] = _positive_probability(model, X_modified)

    return grid, curves, chosen

def compute_ale(model, X, feature, n_bins=20):
    x = X[feature].to_numpy(dtype=float)
    quantiles = np.linspace(0, 1, n_bins + 1)
    edges = np.quantile(x, quantiles)
    edges = np.unique(edges)

    if len(edges) < 3:
        raise ValueError(f"Not enough unique values to create ALE bins for '{feature}'.")

    centers = []
    effects = []
    counts = []

    for lo, hi in zip(edges[:-1], edges[1:]):
        if hi <= lo:
            continue

        # Include the right edge in the final interval.
        mask = (x >= lo) & ((x < hi) if hi < edges[-1] else (x <= hi))
        idx = np.flatnonzero(mask)

        if len(idx) == 0:
            continue

        X_lo = X.iloc[idx].copy()
        X_hi = X.iloc[idx].copy()
        X_lo[feature] = lo
        X_hi[feature] = hi

        diff = _positive_probability(model, X_hi) - _positive_probability(model, X_lo)

        centers.append((lo + hi) / 2.0)
        effects.append(float(np.mean(diff)))
        counts.append(int(len(idx)))

    centers = np.asarray(centers)
    effects = np.asarray(effects)

    accumulated = np.cumsum(effects)
    ale = accumulated - np.average(accumulated, weights=np.asarray(counts))

    return centers, ale, np.asarray(counts), edges
