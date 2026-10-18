"""
bench.py - benchmark suite for the river-level alerting model.

Usage from the notebook (hourly = DataFrame with columns level, rain, hourly UTC index):

    from bench import run_cv, reliability_table
    summary, per_fold, scores = run_cv(hourly)
    print(summary.to_string())

What it does
- Leave-one-water-year-out folds (Oct-Sep). Each fold: the high-level threshold, the
  alert thresholds and every model are chosen/fitted on training years only.
- Models: two simple rules, logistic regression, HistGradientBoosting, RandomForest,
  RandomForest + Platt (sigmoid) calibration.
- Three operating points per model, picked on a validation year inside the training years:
  high_recall (recall >= min_recall), balanced (max F1), high_precision (precision >= min_precision).
- Event-level results on the held-out year: events caught, lead time, false alerts per month.
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
from sklearn.calibration import calibration_curve
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

H = pd.Timedelta(hours=1)
FEATURE_WINDOW_H = 168  # longest rolling window in build_features


# ---------------------------------------------------------------- features / dataset
def build_features(df, calendar=False):
    f = pd.DataFrame(index=df.index)
    f["level"] = df.level
    for h in (1, 3, 6, 12, 24):
        f[f"level_chg_{h}h"] = df.level.diff(h)
    for h in (6, 24, 72):
        f[f"level_max_{h}h"] = df.level.rolling(h).max()
    f["level_mean_72h"] = df.level.rolling(72).mean()
    for h in (1, 3, 6, 12, 24, 72, 168):
        f[f"rain_{h}h"] = df.rain.rolling(h, min_periods=1).sum()
    if calendar:
        f["hour"] = df.index.hour
        f["month"] = df.index.month
    return f


def future_max(level, h):
    """Max level over the NEXT h hours (t+1 ... t+h)."""
    return level.iloc[::-1].rolling(h, min_periods=h).max().iloc[::-1].shift(-1)


def make_dataset(hourly, level_threshold, horizon, calendar=False):
    feat = build_features(hourly, calendar)
    fut = future_max(hourly.level, horizon)
    data = feat.assign(target=(fut >= level_threshold).astype(int))
    return data[(hourly.level < level_threshold) & fut.notna()].dropna()


def water_year(index):
    return np.asarray(index.year + (index.month >= 10))


def region_mask(index, lo, hi, before, after):
    """True for rows inside [lo - before, hi + after]."""
    return (index >= lo - before) & (index <= hi + after)


# ---------------------------------------------------------------- models
class RuleScore:
    """A rule used as a score: alert when column >= threshold (threshold chosen like any other)."""
    def __init__(self, col):
        self.col = col
    def fit(self, X, y):
        return self
    def score(self, X):
        return X[self.col].to_numpy(dtype=float)


class Sk:
    def __init__(self, est):
        self.est = est
    def fit(self, X, y):
        self.est.fit(X, y)
        return self
    def score(self, X):
        return self.est.predict_proba(X)[:, 1]


class Platt:
    """Sigmoid calibration of a score, fitted on a validation slice."""
    def __init__(self):
        self.lr = LogisticRegression(C=1e6, max_iter=1000)
    @staticmethod
    def _z(p):
        p = np.clip(p, 1e-4, 1 - 1e-4)
        return np.log(p / (1 - p)).reshape(-1, 1)
    def fit(self, p, y):
        self.lr.fit(self._z(p), y)
        return self
    def transform(self, p):
        return self.lr.predict_proba(self._z(p))[:, 1]


def default_models(seed=42):
    return {
        "rule_rise_3h": lambda: RuleScore("level_chg_3h"),
        "rule_rain_24h": lambda: RuleScore("rain_24h"),
        "logreg": lambda: Sk(make_pipeline(StandardScaler(),
                             LogisticRegression(class_weight="balanced", max_iter=2000))),
        "hist_gb": lambda: Sk(HistGradientBoostingClassifier(
            max_depth=3, learning_rate=0.05, max_iter=200, random_state=seed)),
        "rf": lambda: Sk(RandomForestClassifier(
            n_estimators=300, min_samples_leaf=5, class_weight="balanced_subsample",
            n_jobs=-1, random_state=seed)),
    }

PROB_MODELS = {"logreg", "hist_gb", "rf", "rf_platt"}


# ---------------------------------------------------------------- operating points
def pick_thresholds(y, s, min_recall=0.8, min_precision=0.5):
    """Return {op_name: threshold} chosen on (y, s). Responders care about misses, so
    high_recall takes the highest threshold that still reaches min_recall."""
    y, s = np.asarray(y), np.asarray(s, dtype=float)
    cands = np.unique(np.quantile(s, np.linspace(0.5, 0.999, 250)))
    rows = []
    for t in cands:
        pred = s >= t
        tp = int((pred & (y == 1)).sum()); fp = int((pred & (y == 0)).sum()); fn = int(((~pred) & (y == 1)).sum())
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        rows.append((t, prec, rec, f1))
    r = np.array(rows)
    ok_rec = r[r[:, 2] >= min_recall]
    t_rec = ok_rec[:, 0].max() if len(ok_rec) else r[np.argmax(r[:, 2]), 0]
    ok_prec = r[(r[:, 1] >= min_precision) & (r[:, 2] > 0)]
    t_prec = ok_prec[:, 0].min() if len(ok_prec) else r[np.argmax(r[:, 1]), 0]
    t_bal = r[np.argmax(r[:, 3]), 0]
    return {"high_recall": float(t_rec), "balanced": float(t_bal), "high_precision": float(t_prec)}


# ---------------------------------------------------------------- events
def find_events(level, thr, merge_h=12):
    """Start time of each event = first hour of a run at/above thr; runs closer than merge_h merge."""
    t = level.index[(level >= thr).to_numpy()]
    if len(t) == 0:
        return []
    starts, last = [t[0]], t[0]
    for x in t[1:]:
        if (x - last) / H > merge_h:
            starts.append(x)
        last = x
    return starts


def alert_episodes(alert_times, cooldown_h):
    eps = []
    for t in sorted(alert_times):
        if eps and (t - eps[-1][1]) / H <= cooldown_h:
            eps[-1][1] = t
        else:
            eps.append([t, t])
    return eps


def event_metrics(alert_times, event_starts, horizon, cooldown_h):
    h = pd.Timedelta(hours=horizon)
    alert_times = sorted(alert_times)
    caught, leads = 0, []
    for e in event_starts:
        a = [t for t in alert_times if e - h <= t < e]
        if a:
            caught += 1
            leads.append((e - min(a)) / H)
    eps = alert_episodes(alert_times, cooldown_h)
    false_eps = sum(1 for s, l in eps if not any(s < e <= l + h for e in event_starts))
    return {"events": len(event_starts), "caught": caught, "leads": leads,
            "episodes": len(eps), "false_episodes": false_eps}


# ---------------------------------------------------------------- cross-validation
def run_cv(hourly, horizon=24, pct=0.95, models=None, min_recall=0.8, min_precision=0.5,
           merge_h=12, cooldown_h=12, calendar=False, min_year_rows=2000, seed=42, verbose=True):
    """Leave-one-water-year-out evaluation. Returns (summary, per_fold, scores)."""
    models = models or default_models(seed)
    gap = pd.Timedelta(hours=horizon)
    after = pd.Timedelta(hours=FEATURE_WINDOW_H)
    lvl = hourly.level
    wy = pd.Series(water_year(hourly.index), index=hourly.index)
    years = sorted(y for y, c in wy.value_counts().items() if c >= min_year_rows)
    if len(years) < 2:
        raise ValueError(f"Need at least 2 usable water years, found {years}")

    records, scores = [], {}
    for test_y in years:
        test_hours = hourly.index[(wy == test_y).to_numpy()]
        t0, t1 = test_hours[0], test_hours[-1]
        train_years = [y for y in years if y != test_y]
        val_y = train_years[-1]
        val_hours = hourly.index[(wy == val_y).to_numpy()]
        v0, v1 = val_hours[0], val_hours[-1]

        # thresholds from TRAINING years only
        train_lvl = lvl[(wy.isin(train_years)).to_numpy() & ~region_mask(lvl.index, t0, t1, gap, after)]
        thr = float(train_lvl.quantile(pct))
        data = make_dataset(hourly, thr, horizon, calendar)
        feats = [c for c in data.columns if c != "target"]

        in_test = (data.index >= t0) & (data.index <= t1)
        in_train = data.index.isin(hourly.index[wy.isin(train_years).to_numpy()]) \
            & ~region_mask(data.index, t0, t1, gap, after)
        in_val = in_train & (data.index >= v0) & (data.index <= v1)
        in_inner = in_train & ~region_mask(data.index, v0, v1, gap, after)
        test, train, val, inner = data[in_test], data[in_train], data[in_val], data[in_inner]

        if inner.target.sum() < 5 or train.target.sum() < 5 or test.empty:
            if verbose:
                print(f"[fold WY{test_y}] skipped: too few positive rows "
                      f"(inner={int(inner.target.sum())}, train={int(train.target.sum())})")
            continue
        events = find_events(lvl[(lvl.index >= t0) & (lvl.index <= t1)], thr, merge_h)
        test_days = len(test_hours) / 24
        if verbose:
            print(f"[fold WY{test_y}] thr={thr:.3f} m | train rows={len(train)} | test rows={len(test)} "
                  f"| test events={len(events)} | val positives={int(val.target.sum())}")

        def evaluate(name, st, ops):
            for op, t in ops.items():
                alerts = test.index[st >= t]
                m = event_metrics(alerts, events, horizon, cooldown_h)
                records.append({"fold": f"WY{test_y}", "model": name, "op": op, "threshold": t,
                                "test_days": test_days, **m})
            auc = roc_auc_score(test.target, st) if test.target.nunique() > 1 else np.nan
            pr = average_precision_score(test.target, st) if test.target.sum() > 0 else np.nan
            records.append({"fold": f"WY{test_y}", "model": name, "op": "_rowmetrics",
                            "roc_auc": auc, "pr_auc": pr,
                            "brier": brier_score_loss(test.target, np.clip(st, 0, 1)) if name in PROB_MODELS else np.nan})
            scores.setdefault(name, []).append((test.target.to_numpy(), st))

        fitted_rf_inner = None
        for name, make in models.items():
            m_in = make().fit(inner[feats], inner.target)
            sv = m_in.score(val[feats])
            ops = pick_thresholds(val.target, sv, min_recall, min_precision) if val.target.sum() > 0 \
                else {k: float(np.quantile(sv, 0.95)) for k in ("high_recall", "balanced", "high_precision")}
            m_full = make().fit(train[feats], train.target)
            evaluate(name, m_full.score(test[feats]), ops)
            if name == "rf":
                fitted_rf_inner = (m_in, sv)

        if fitted_rf_inner is not None and val.target.sum() > 0:
            m_in, sv = fitted_rf_inner
            platt = Platt().fit(sv, val.target)
            sv_c = platt.transform(sv)
            ops_c = pick_thresholds(val.target, sv_c, min_recall, min_precision)
            evaluate("rf_platt", platt.transform(m_in.score(test[feats])), ops_c)

    per_fold = pd.DataFrame(records)
    return summarise(per_fold), per_fold, scores


def summarise(per_fold):
    ev = per_fold[per_fold.op != "_rowmetrics"]
    out = []
    for (model, op), g in ev.groupby(["model", "op"], sort=False):
        leads = [x for l in g["leads"] for x in l]
        out.append({"model": model, "op": op,
                    "events_caught": f"{int(g.caught.sum())}/{int(g.events.sum())}",
                    "median_lead_h": float(np.median(leads)) if leads else np.nan,
                    "false_alerts_per_month": g.false_episodes.sum() / (g.test_days.sum() / 30.44)})
    summary = pd.DataFrame(out)
    rm = per_fold[per_fold.op == "_rowmetrics"].groupby("model", sort=False)[["roc_auc", "pr_auc", "brier"]].mean()
    return summary.merge(rm, left_on="model", right_index=True, how="left").round(3)


def reliability_table(scores, model, bins=10):
    """Pooled reliability data for a probabilistic model (plot frac_pos vs mean_pred)."""
    y = np.concatenate([a for a, _ in scores[model]])
    p = np.concatenate([b for _, b in scores[model]])
    frac, mean = calibration_curve(y, p, n_bins=bins, strategy="quantile")
    return pd.DataFrame({"mean_pred": mean, "frac_pos": frac})


# ---------------------------------------------------------------- synthetic demo
def synthetic_hourly(years=3, seed=0):
    rng = np.random.default_rng(seed)
    n = 24 * 365 * years
    idx = pd.date_range("2022-10-01", periods=n, freq="1h", tz="UTC")
    rain = np.zeros(n)
    for s in rng.integers(0, n - 60, int(years * 70)):
        month = idx[s].month
        w = 1.8 if month in (10, 11, 12, 1, 2, 3) else 0.6
        dur = int(rng.integers(3, 30))
        rain[s:s + dur] += rng.gamma(1.2, 0.5 * w, dur)
    fast = slow = 0.0
    level = np.zeros(n)
    for i in range(n):
        fast = 0.85 * fast + rain[i]
        slow = 0.995 * slow + 0.03 * rain[i]
        level[i] = 0.3 + 0.07 * fast + 0.6 * slow + rng.normal(0, 0.01)
    return pd.DataFrame({"level": level, "rain": rain}, index=idx)


if __name__ == "__main__":
    df = synthetic_hourly()
    summary, per_fold, scores = run_cv(df)
    print()
    print(summary.to_string(index=False))
    print()
    print(reliability_table(scores, "rf").round(3).to_string(index=False))
    print(reliability_table(scores, "rf_platt").round(3).to_string(index=False))
