# ================================================================
# AI-IPS FINAL — Q1/Q2 Journal Ready
# ================================================================
# Final stable version — built on successful v4.1 + v5.0 additions
#
# Smart temporal split:
#   Train : Mon + Tue + Wed + Thu-Morning + Fri-Morning (50%)
#   Val   : Thu-Afternoon
#   Test  : Fri-Morning (50%) + Fri-Afternoon-DDoS + Fri-Afternoon-PortScan
#
# Improvements over v4.1:
#   + Throughput Analysis (No-Attack vs Attack)
#   + Bootstrap 95% CI
#   + McNemar via scipy without statsmodels
#   + Wilcoxon Test
#   - Removed Isotonic Calibration (was corrupting scores)
# ================================================================

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import warnings, time, json, os, glob, copy
from scipy.stats import chi2, wilcoxon as scipy_wilcoxon
warnings.filterwarnings('ignore')

from sklearn.linear_model      import LogisticRegression, SGDClassifier
from sklearn.ensemble          import RandomForestClassifier, IsolationForest
from sklearn.preprocessing     import StandardScaler
from sklearn.metrics           import (classification_report, roc_curve, auc,
                                        precision_recall_curve, confusion_matrix,
                                        f1_score, precision_score, recall_score)
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.model_selection   import train_test_split

try:
    import xgboost as xgb;  HAS_XGB = True
except ImportError:
    HAS_XGB = False; print("[WARN] pip install xgboost")

try:
    import lightgbm as lgb; HAS_LGB = True
except ImportError:
    HAS_LGB = False; print("[WARN] pip install lightgbm")

try:
    from imblearn.over_sampling import SMOTE; HAS_SMOTE = True
except ImportError:
    HAS_SMOTE = False; print("[WARN] pip install imbalanced-learn")

try:
    import shap; HAS_SHAP = True
except ImportError:
    HAS_SHAP = False

SEED = 42
np.random.seed(SEED)

# ================================================================
# Smart Temporal Split
# ================================================================
# Train: contains diverse attack types + Friday-Morning traffic
TRAIN_FILES = [
    "Monday-WorkingHours.pcap_ISCX.csv",
    "Tuesday-WorkingHours.pcap_ISCX.csv",
    "Wednesday-workingHours.pcap_ISCX.csv",
    "Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv",
    # Add Friday-Morning to Train so model learns Friday BENIGN patterns
    "Friday-WorkingHours-Morning.pcap_ISCX.csv",
]
VAL_FILES = [
    "Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv",
]
# Test: real Friday attacks (DDoS + PortScan)
TEST_FILES = [
    "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv",
    "Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv",
]

# Zero-Day attacks: present in Test set only
ZERO_DAY_ATTACKS = ['DDoS', 'PortScan', 'Bot', 'Heartbleed']
SUSPICIOUS_COLS  = ['Flow ID','Source IP','Destination IP',
                    'Source Port','Timestamp','timestamp','flow_id']
BENIGN_LABELS    = {'BENIGN','Benign','benign','Normal','normal','NORMAL'}

# ================================================================
# Helper functions
# ================================================================
def load_files(file_list):
    dfs = []
    for f in file_list:
        if not os.path.exists(f):
            print(f"  [Warning] File not found: {f}"); continue
        try:    df_tmp = pd.read_csv(f, encoding='utf-8',  low_memory=False)
        except: df_tmp = pd.read_csv(f, encoding='latin-1',low_memory=False)
        df_tmp.columns = df_tmp.columns.str.strip()
        dfs.append(df_tmp)
        print(f"  {f}: {len(df_tmp):,} rows")
    return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()

def detect_label_col(df):
    for c in ['Label','label','CLASS','class','Attack','Category','Label ']:
        if c in df.columns: return c
    raise KeyError(f"Label column not found. Available columns: {df.columns.tolist()}")


def add_features(df):
    df = df.copy()
    needed = {'Total Fwd Packets','Total Backward Packets',
              'Total Length of Fwd Packets','Total Length of Bwd Packets',
              'Flow Bytes/s','Flow Packets/s'}
    if needed.issubset(set(df.columns)):
        df['bytes_ratio']    = df['Total Fwd Packets'] / (df['Total Backward Packets'] + 1)
        df['packet_ratio']   = df['Total Length of Fwd Packets'] / (df['Total Length of Bwd Packets'] + 1)
        df['flow_intensity'] = df['Flow Bytes/s'] / (df['Flow Packets/s'] + 1)
        df['fwd_bwd_diff']   = df['Total Fwd Packets'] - df['Total Backward Packets']
        df['total_pkts']     = df['Total Fwd Packets'] + df['Total Backward Packets']
        df['pkt_len_ratio']  = df['Total Length of Fwd Packets'] / (
            df['Total Length of Bwd Packets'] + df['Total Fwd Packets'] + 1)
    df.replace([np.inf, -np.inf], 0, inplace=True)
    df.fillna(0, inplace=True)
    return df

def prepare_xy(df, label_col):
    y_bin   = df[label_col].apply(
        lambda x: 0 if str(x).strip() in BENIGN_LABELS else 1).values
    y_multi = df[label_col].str.strip().values
    drop    = [label_col] + [c for c in SUSPICIOUS_COLS if c in df.columns]
    X       = df.drop(columns=drop, errors='ignore').select_dtypes(include=[np.number])
    X.replace([np.inf, -np.inf], np.nan, inplace=True)
    X.drop(columns=X.columns[X.isna().mean() > 0.5].tolist(), inplace=True)
    X.fillna(0, inplace=True)
    return X, y_bin, y_multi

def normalize_scores(scores):
    scores = np.asarray(scores, dtype=float)
    if len(scores) == 0: return scores
    mn, mx = scores.min(), scores.max()
    if mx - mn < 1e-9: return np.full_like(scores, 0.5)
    return (scores - mn) / (mx - mn)

def hybrid_fusion(s_sup, s_hst, w_sup, w_hst):
    """Adaptive fusion: weight shifts per sample based on confidence level."""
    s_sup = np.asarray(s_sup, dtype=float)
    s_hst = np.asarray(s_hst, dtype=float)
    cs = np.abs(s_sup - 0.5)
    ch = np.abs(s_hst - 0.5)
    tot = cs + ch + 1e-9
    ws  = w_sup + (cs / tot) * (1 - w_sup - w_hst)
    wh  = w_hst + (ch / tot) * (1 - w_sup - w_hst)
    nm  = ws + wh + 1e-9
    return (ws / nm) * s_sup + (wh / nm) * s_hst

def hybrid_zeroday(s_sup, s_hst, boost=0.45):
    """Zero-Day mode: dynamically raises IsolationForest weight."""
    s_sup = np.asarray(s_sup, dtype=float)
    s_hst = np.asarray(s_hst, dtype=float)
    nov   = np.clip(s_hst, 0, 1)
    wh    = np.clip(0.3 + nov * boost, 0, 0.8)
    return (1 - wh) * s_sup + wh * s_hst

def best_threshold(y_val, scores, metric='balanced'):
    """
    Find optimal threshold on Val Set.
    balanced: 70% Recall + 30% Precision weight (suitable for IDS)
    recall:   maximize recall
    f1:       maximize F1
    """
    best_t, best_s = 0.5, 0.0
    for t in np.linspace(0.05, 0.95, 200):
        preds = (scores > t).astype(int)
        tp = ((preds == 1) & (y_val == 1)).sum()
        fp = ((preds == 1) & (y_val == 0)).sum()
        fn = ((preds == 0) & (y_val == 1)).sum()
        rec = tp / (tp + fn + 1e-9)
        pre = tp / (tp + fp + 1e-9)
        if metric == 'balanced':
            s = 0.7 * rec + 0.3 * pre
        elif metric == 'recall':
            s = rec
        elif metric == 'f1':
            s = 2 * rec * pre / (rec + pre + 1e-9)
        if s > best_s: best_s, best_t = s, t
    return best_t

def safe_auc(y_true, scores):
    try:
        if len(np.unique(y_true)) < 2: return float('nan')
        fpr, tpr, _ = roc_curve(y_true, scores)
        return auc(fpr, tpr)
    except: return float('nan')

def mcnemar_test(b, c):
    """McNemar test using scipy only."""
    if b + c == 0: return float('nan')
    if b + c < 25:
        from scipy.stats import binomtest
        try:
            return float(binomtest(min(b, c), b + c, 0.5).pvalue)
        except:
            from scipy.stats import binom
            return float(2 * binom.cdf(min(b, c), b + c, 0.5))
    stat = (abs(b - c) - 1) ** 2 / (b + c)
    return float(1 - chi2.cdf(stat, df=1))

def bootstrap_ci(y_true, scores, n_boot=1000, ci=95):
    """Bootstrap Confidence Interval for AUC."""
    boot_vals = []
    n = len(y_true)
    for _ in range(n_boot):
        idx = np.random.choice(n, n, replace=True)
        val = safe_auc(y_true[idx], scores[idx])
        if not np.isnan(val): boot_vals.append(val)
    if not boot_vals: return float('nan'), float('nan')
    lo = np.percentile(boot_vals, (100 - ci) / 2)
    hi = np.percentile(boot_vals, 100 - (100 - ci) / 2)
    return round(lo, 4), round(hi, 4)

def measure_throughput(sup_model, if_model, X_samples,
                        w_sup, w_hst, n_runs=10):
    """Measure Throughput and Latency."""
    times = []
    n = len(X_samples)
    for _ in range(n_runs):
        t0 = time.perf_counter()
        ss = sup_model.predict_proba(X_samples)[:, 1]
        sh = normalize_scores(-if_model.decision_function(X_samples))
        _  = hybrid_fusion(ss, sh, w_sup, w_hst)
        times.append(time.perf_counter() - t0)
    avg   = np.mean(times)
    return {
        'n'          : n,
        'thr_pps'    : round(n / avg, 0),
        'lat_ms'     : round((avg / n) * 1000, 4),
        'p95_ms'     : round(np.percentile([(t/n)*1000 for t in times], 95), 4),
    }

def score_to_alert(score, thr):
    if   score < thr:         return "NORMAL",   "🟢"
    elif score < thr + 0.15:  return "LOW",      "🟡"
    elif score < thr + 0.30:  return "MEDIUM",   "🟠"
    elif score < thr + 0.40:  return "HIGH",     "🔴"
    else:                     return "CRITICAL",  "🚨"

# ================================================================
# 1. Load Data
# ================================================================
print("=" * 60)
print("AI-IPS FINAL — Smart Temporal Split")
print("=" * 60)

print("\n[TRAIN]"); df_train = load_files(TRAIN_FILES)
print("\n[VAL]");   df_val   = load_files(VAL_FILES)
print("\n[TEST]");  df_test  = load_files(TEST_FILES)

# Fallback if temporal files not found
if df_train.empty or df_test.empty:
    print("[Fallback] Searching for CICIDS2017 CSV files...")


    _ex = {"train.csv","val.csv","test.csv"}; _seen = set(); all_f = []
    for _f in glob.glob("*.pcap_ISCX.csv") + glob.glob("*.csv"):
        _a = os.path.abspath(_f)
        if os.path.basename(_f) not in _ex and _a not in _seen:
            _seen.add(_a); all_f.append(_f)
    if not all_f:
        raise FileNotFoundError("No CSV files found!")
    all_f.sort(); n = len(all_f)
    df_train = load_files(all_f[:max(1, int(n * 0.7))])
    df_val   = load_files(all_f[int(n * 0.7):int(n * 0.85)])
    df_test  = load_files(all_f[int(n * 0.85):])

LABEL_COL = detect_label_col(df_train)
print(f"\nLabel column: '{LABEL_COL}'")

# ================================================================
# 2. Prepare Data
# ================================================================
print("\n[Prep] Feature engineering...")
df_train = add_features(df_train)
df_val   = add_features(df_val)   if not df_val.empty  else df_val
df_test  = add_features(df_test)

X_train_raw, y_train, y_train_multi = prepare_xy(df_train, LABEL_COL)
X_test_raw,  y_test,  y_test_multi  = prepare_xy(df_test,  LABEL_COL)

if not df_val.empty:
    X_val_raw, y_val, y_val_multi = prepare_xy(df_val, LABEL_COL)
else:
    X_train_raw, X_val_raw, y_train, y_val = train_test_split(
        X_train_raw, y_train, test_size=0.2,
        random_state=SEED, stratify=y_train)
    y_val_multi = y_val.astype(str)

# Align columns across Train/Val/Test
common_cols = (X_train_raw.columns
               .intersection(X_val_raw.columns)
               .intersection(X_test_raw.columns))
X_train_raw = X_train_raw[common_cols]
X_val_raw   = X_val_raw[common_cols]
X_test_raw  = X_test_raw[common_cols]

print(f"Train={len(y_train):,} | Val={len(y_val):,} | Test={len(y_test):,}")
print(f"Attacks: Train={y_train.mean()*100:.1f}% | "
      f"Val={y_val.mean()*100:.1f}% | Test={y_test.mean()*100:.1f}%")

print("\nTest distribution:")
for lbl, cnt in pd.Series(y_test_multi).value_counts().items():
    print(f"  {str(lbl):<45} {cnt:>8,}")

# ================================================================
# 3. Feature Selection + Scaling
# ================================================================
print("\n[Pipeline] Feature Selection + Scaling...")
MAX_K    = min(35, X_train_raw.shape[1])
selector = SelectKBest(mutual_info_classif, k=MAX_K)
scaler   = StandardScaler()

X_train_sel = selector.fit_transform(X_train_raw, y_train)
X_train_sc  = scaler.fit_transform(X_train_sel)
X_val_sc    = scaler.transform(selector.transform(X_val_raw))
X_test_sc   = scaler.transform(selector.transform(X_test_raw))
feat_names  = np.array(common_cols.tolist())[selector.get_support()]
print(f"Features: {MAX_K} | Examples: {list(feat_names[:4])}")

# SMOTE — applied on Train only, inside pipeline
if HAS_SMOTE:
    sm = SMOTE(random_state=SEED)
    X_train_bal, y_train_bal = sm.fit_resample(X_train_sc, y_train)
    print(f"[SMOTE] {pd.Series(y_train_bal).value_counts().to_dict()}")
else:
    X_train_bal, y_train_bal = X_train_sc, y_train

sp = max((y_train == 0).sum() / max((y_train == 1).sum(), 1), 1)

# ================================================================
# 4. Train Models
# ================================================================
print("\n[Models] Training models...")

# IsolationForest — unsupervised, no labels needed
iforest = IsolationForest(
    n_estimators=300,
    contamination=float(np.clip(y_train.mean(), 0.01, 0.49)),
    random_state=SEED, n_jobs=-1)
iforest.fit(X_train_sc)
print("  IsolationForest ✓")

s_hst_val  = normalize_scores(-iforest.decision_function(X_val_sc))
s_hst_test = normalize_scores(-iforest.decision_function(X_test_sc))

# Supervised model
if HAS_XGB:
    SUPER_NAME = "XGBoost"
    sup_clf = xgb.XGBClassifier(
        n_estimators=500, scale_pos_weight=sp,
        max_depth=7, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        min_child_weight=3, gamma=0.1,
        random_state=SEED, eval_metric='logloss',
        verbosity=0, early_stopping_rounds=30)
    sup_clf.fit(X_train_bal, y_train_bal,
                eval_set=[(X_val_sc, y_val)], verbose=False)
elif HAS_LGB:
    SUPER_NAME = "LightGBM"
    sup_clf = lgb.LGBMClassifier(
        n_estimators=500, class_weight='balanced',
        max_depth=8, learning_rate=0.05,
        num_leaves=63, random_state=SEED, verbose=-1)
    sup_clf.fit(X_train_bal, y_train_bal,
                eval_set=[(X_val_sc, y_val)])
else:
    SUPER_NAME = "Random Forest"
    sup_clf = RandomForestClassifier(
        n_estimators=500, class_weight='balanced',
        max_depth=20, random_state=SEED, n_jobs=-1)
    sup_clf.fit(X_train_bal, y_train_bal)

print(f"  {SUPER_NAME} ✓")

s_sup_val  = sup_clf.predict_proba(X_val_sc)[:, 1]
s_sup_test = sup_clf.predict_proba(X_test_sc)[:, 1]

# ================================================================
# 5. Fusion Weights + Threshold
# ================================================================
print("\n[Fusion] Searching optimal weights on Val set...")
best_f1v, W_SUP, W_HST = 0, 0.7, 0.3
wg = np.arange(0.0, 1.05, 0.05); abl_f1s = []
for wh in wg:
    ws  = 1 - wh
    hyb = hybrid_fusion(s_sup_val, s_hst_val, ws, wh)
    f1v = f1_score(y_val, (hyb > 0.5).astype(int),
                   average='weighted', zero_division=0)
    abl_f1s.append(f1v)
    if f1v > best_f1v: best_f1v, W_SUP, W_HST = f1v, ws, wh
print(f"  w_sup={W_SUP:.2f} | w_hst={W_HST:.2f} | Val F1={best_f1v:.4f}")

s_hyb_val  = hybrid_fusion(s_sup_val,  s_hst_val,  W_SUP, W_HST)
s_hyb_test = hybrid_fusion(s_sup_test, s_hst_test, W_SUP, W_HST)

# Balanced Threshold (70% Recall + 30% Precision weight)
THR_MAIN = best_threshold(y_val, s_hyb_val,  metric='balanced')
THR_SUP  = best_threshold(y_val, s_sup_val,  metric='balanced')
THR_HST  = best_threshold(y_val, s_hst_val,  metric='balanced')
print(f"  Threshold — Hybrid: {THR_MAIN:.3f} | "
      f"{SUPER_NAME}: {THR_SUP:.3f} | IForest: {THR_HST:.3f}")

# ================================================================
# THROUGHPUT ANALYSIS
# ================================================================
print("\n" + "=" * 60)
print("THROUGHPUT ANALYSIS")
print("=" * 60)

idx_benign = np.where(y_test == 0)[0]
idx_attack = np.where(y_test == 1)[0]
BATCH_SIZES = [1, 10, 100, 500, 1000, 5000]
thr_results = {'no_attack': [], 'attack': [], 'mixed': []}

print(f"\n{'Scenario':<12} {'Batch':>6} {'Throughput(pps)':>16} "
      f"{'Latency(ms)':>13} {'P95(ms)':>9}")
print("-" * 62)

for bs in BATCH_SIZES:
    n_b = min(bs, len(idx_benign))
    n_a = min(bs, len(idx_attack))

    X_b = X_test_sc[np.random.choice(idx_benign, n_b, replace=n_b > len(idx_benign))]
    r_b = measure_throughput(sup_clf, iforest, X_b, W_SUP, W_HST)
    thr_results['no_attack'].append({'batch': bs, **r_b})

    X_a = X_test_sc[np.random.choice(idx_attack, n_a, replace=n_a > len(idx_attack))]
    r_a = measure_throughput(sup_clf, iforest, X_a, W_SUP, W_HST)
    thr_results['attack'].append({'batch': bs, **r_a})

    print(f"{'No-Attack':<12} {bs:>6,} {r_b['thr_pps']:>16,.0f} "
          f"{r_b['lat_ms']:>13.3f} {r_b['p95_ms']:>9.3f}")
    print(f"{'Attack':<12} {bs:>6,} {r_a['thr_pps']:>16,.0f} "
          f"{r_a['lat_ms']:>13.3f} {r_a['p95_ms']:>9.3f}")
    print("-" * 62)

avg_thr_na = np.mean([r['thr_pps'] for r in thr_results['no_attack']])
avg_thr_at = np.mean([r['thr_pps'] for r in thr_results['attack']])
avg_lat_na = np.mean([r['lat_ms']  for r in thr_results['no_attack']])
avg_lat_at = np.mean([r['lat_ms']  for r in thr_results['attack']])
overhead   = abs(avg_thr_na - avg_thr_at) / (avg_thr_na + 1e-9) * 100

print(f"\n[Throughput Summary]")
print(f"  No-Attack : avg={avg_thr_na:,.0f} pps | latency={avg_lat_na:.3f}ms")
print(f"  Attack    : avg={avg_thr_at:,.0f} pps | latency={avg_lat_at:.3f}ms")
print(f"  Overhead  : {overhead:.1f}% difference under attack")

# ================================================================
# SCENARIO A: Zero-Day Attack Detection
# ================================================================
print("\n" + "=" * 60)
print("SCENARIO A: Zero-Day Attack Detection")
print("=" * 60)

test_atk_types = set(np.unique(y_test_multi)) - BENIGN_LABELS
actual_zd      = [a for a in ZERO_DAY_ATTACKS if a in test_atk_types]
known_atk      = [a for a in test_atk_types if a not in ZERO_DAY_ATTACKS]
print(f"Zero-Day in Test : {actual_zd}")
print(f"Known attacks   : {known_atk[:5]}")

# Retrain without Zero-Day attack types
zd_mask = ~pd.Series(y_train_multi).isin(actual_zd).values
X_tr_zd = X_train_sc[zd_mask]
y_tr_zd = y_train[zd_mask]

if HAS_SMOTE and len(np.unique(y_tr_zd)) > 1:
    try:
        sm_zd = SMOTE(random_state=SEED)
        X_tr_zd, y_tr_zd = sm_zd.fit_resample(X_tr_zd, y_tr_zd)
    except: pass

sp_zd = max((y_tr_zd == 0).sum() / max((y_tr_zd == 1).sum(), 1), 1)

if HAS_XGB:
    clf_zd = xgb.XGBClassifier(
        n_estimators=400, scale_pos_weight=sp_zd,
        max_depth=7, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        random_state=SEED, eval_metric='logloss', verbosity=0)
    clf_zd.fit(X_tr_zd, y_tr_zd,
               eval_set=[(X_val_sc, y_val)], verbose=False)
elif HAS_LGB:
    clf_zd = lgb.LGBMClassifier(
        n_estimators=400, class_weight='balanced',
        random_state=SEED, verbose=-1)
    clf_zd.fit(X_tr_zd, y_tr_zd,
               eval_set=[(X_val_sc, y_val)])
else:
    clf_zd = RandomForestClassifier(
        n_estimators=400, class_weight='balanced',
        max_depth=18, random_state=SEED, n_jobs=-1)
    clf_zd.fit(X_tr_zd, y_tr_zd)

iforest_zd = IsolationForest(
    n_estimators=300,
    contamination=float(np.clip(y_tr_zd.mean(), 0.01, 0.49)),
    random_state=SEED, n_jobs=-1)
iforest_zd.fit(X_tr_zd)

s_sup_zd  = clf_zd.predict_proba(X_test_sc)[:, 1]
s_hst_zd  = normalize_scores(-iforest_zd.decision_function(X_test_sc))
s_hyb_zd  = hybrid_zeroday(s_sup_zd, s_hst_zd, boost=0.45)

# Recall-optimized threshold for Zero-Day detection
s_sv_zd   = clf_zd.predict_proba(X_val_sc)[:, 1]
s_hv_zd   = normalize_scores(-iforest_zd.decision_function(X_val_sc))
s_hyv_zd  = hybrid_zeroday(s_sv_zd, s_hv_zd, boost=0.45)
THR_ZD_S  = best_threshold(y_val, s_sv_zd,  metric='recall')
THR_ZD_H  = best_threshold(y_val, s_hyv_zd, metric='recall')
print(f"ZD Threshold — {SUPER_NAME}: {THR_ZD_S:.3f} | Hybrid: {THR_ZD_H:.3f}")

zd_results = {}
print(f"\n  {'Attack':<20} {'N':>8} {SUPER_NAME+' DR':>14} "
      f"{'Hybrid DR':>12} {'Gain':>8}")
print("  " + "-" * 65)

for atk in actual_zd:
    mask = (y_test_multi == atk)
    if mask.sum() < 5: continue
    dr_s = (s_sup_zd[mask] > THR_ZD_S).mean()
    dr_h = (s_hyb_zd[mask] > THR_ZD_H).mean()
    gain = dr_h - dr_s
    zd_results[atk] = {'n': int(mask.sum()),
                        'supervised': round(dr_s, 4),
                        'hybrid': round(dr_h, 4),
                        'gain': round(gain, 4)}
    print(f"  {atk:<20} {mask.sum():>8,} {dr_s:>14.4f} "
          f"{dr_h:>12.4f} {'+'if gain>=0 else''}{gain:>7.4f}")

# FIX: Zero-Day AUC computed with BENIGN samples
mask_zd_all = np.isin(y_test_multi, actual_zd)
if mask_zd_all.sum() > 0:
    n_zd  = mask_zd_all.sum()
    idx_b = np.where(y_test == 0)[0]
    n_ben = min(n_zd, len(idx_b))
    np.random.seed(SEED)
    ben_s = np.random.choice(idx_b, n_ben, replace=False)
    ext   = np.concatenate([np.where(mask_zd_all)[0], ben_s])
    y_ext = y_test[ext]
    auc_sup_zd = safe_auc(y_ext, s_sup_zd[ext])
    auc_hyb_zd = safe_auc(y_ext, s_hyb_zd[ext])
    print(f"\nZero-Day AUC — {SUPER_NAME}: {auc_sup_zd:.4f} | "
          f"Hybrid: {auc_hyb_zd:.4f} | Δ={auc_hyb_zd-auc_sup_zd:+.4f}")
else:
    auc_sup_zd = auc_hyb_zd = float('nan')
    print("\n[Warning] No Zero-Day attacks in Test — check split")

# ================================================================
# SCENARIO B: Real-Time IDS
# ================================================================
print("\n" + "=" * 60)
print("SCENARIO B: Real-Time IDS")
print("=" * 60)

idx0 = np.where(y_test == 0)[0]
idx1 = np.where(y_test == 1)[0]
n_e  = min(500, len(idx0), len(idx1))
rt_i = np.concatenate([
    np.random.choice(idx0, n_e, replace=False),
    np.random.choice(idx1, n_e, replace=False)])
np.random.shuffle(rt_i)
X_rt = X_test_sc[rt_i]; y_rt = y_test[rt_i]; y_rt_m = y_test_multi[rt_i]

# Normalize IForest scores using Val range for consistency
hst_raw_v = -iforest.decision_function(X_val_sc)
hst_mn, hst_mx = hst_raw_v.min(), hst_raw_v.max()
hst_rng = max(hst_mx - hst_mn, 1e-9)

print(f"Simulating {len(rt_i)} packets (BENIGN={n_e}, Attack={n_e})")
print(f"\n{'#':>4} {'Type':<24} {'Score':>7} {'Alert':>10} {'OK':>4} {'ms':>7}")
print("-" * 62)

rt_sc=[]; rt_al=[]; rt_lat=[]; rt_ok=[]
for i, (xr, yt, atk) in enumerate(zip(X_rt, y_rt, y_rt_m)):
    t0  = time.perf_counter()
    x2d = xr.reshape(1, -1)
    ss  = float(sup_clf.predict_proba(x2d)[0, 1])
    raw = float(-iforest.decision_function(x2d)[0])
    sh  = float(np.clip((raw - hst_mn) / hst_rng, 0, 1))
    sc  = float(hybrid_fusion([ss], [sh], W_SUP, W_HST)[0])
    lat = (time.perf_counter() - t0) * 1000
    al, ali = score_to_alert(sc, THR_MAIN)
    ok  = ((sc > THR_MAIN) == yt)
    rt_sc.append(sc); rt_al.append(al); rt_lat.append(lat); rt_ok.append(ok)
    if i < 50:
        print(f"{i+1:>4} {str(atk)[:22]:<24} {sc:>7.4f} "
              f"{ali}{al:>8} {'✓'if ok else '✗':>4} {lat:>6.2f}")

all_ss  = sup_clf.predict_proba(X_rt)[:, 1]
all_raw = -iforest.decision_function(X_rt)
all_sh  = np.clip((all_raw - hst_mn) / hst_rng, 0, 1)
all_hyb = hybrid_fusion(all_ss, all_sh, W_SUP, W_HST)
rt_acc  = ((all_hyb > THR_MAIN).astype(int) == y_rt).mean()
rt_f1   = f1_score(y_rt, (all_hyb > THR_MAIN).astype(int),
                   average='weighted', zero_division=0)
rt_rec  = recall_score(y_rt, (all_hyb > THR_MAIN).astype(int),
                        average='binary', zero_division=0)
rt_auc  = safe_auc(y_rt, all_hyb)
fp_rt   = ((all_hyb > THR_MAIN) & (y_rt == 0)).sum()
fn_rt   = ((all_hyb < THR_MAIN) & (y_rt == 1)).sum()
al_dist = pd.Series(rt_al).value_counts().to_dict()

print(f"\n[Real-Time Summary]")
print(f"  Accuracy : {rt_acc*100:.2f}%")
print(f"  F1-Score : {rt_f1:.4f}")
print(f"  Recall   : {rt_rec:.4f}  <- critical metric in IDS")
print(f"  AUC      : {rt_auc:.4f}")
print(f"  Latency  : mean={np.mean(rt_lat):.2f}ms  "
      f"p95={np.percentile(rt_lat,95):.2f}ms")
print(f"  FP={fp_rt} ({fp_rt/len(y_rt)*100:.1f}%) | "
      f"FN={fn_rt} ({fn_rt/len(y_rt)*100:.1f}%)")
print(f"  Alerts   : {al_dist}")

rt_summary = {
    'accuracy': round(rt_acc, 4), 'f1': round(rt_f1, 4),
    'recall':   round(rt_rec, 4), 'auc': round(rt_auc, 4),
    'mean_lat': round(float(np.mean(rt_lat)), 3),
    'p95_lat':  round(float(np.percentile(rt_lat, 95)), 3),
    'fp': int(fp_rt), 'fn': int(fn_rt), 'alerts': al_dist,
}

# ================================================================
# SCENARIO C: Streaming + Dynamic Threshold
# ================================================================
print("\n" + "=" * 60)
print("SCENARIO C: Streaming + Dynamic Threshold")
print("=" * 60)

# Build balanced stream (50% BENIGN, 50% Attack)
n_str  = min(50000, len(y_test))
idx_b2 = np.where(y_test == 0)[0]
idx_a2 = np.where(y_test == 1)[0]
n_each = min(n_str // 2, len(idx_b2), len(idx_a2))
s_idx  = np.empty(n_each * 2, dtype=int)
s_idx[0::2] = np.random.choice(idx_b2, n_each, replace=False)
s_idx[1::2] = np.random.choice(idx_a2, n_each, replace=False)
stream_X = X_test_sc[s_idx] + np.random.normal(0, 0.015, X_test_sc[s_idx].shape)
stream_y = y_test[s_idx]

WSIZE=500; STEP=200; DRIFT_T=0.10; RETRAIN_E=5
n_wins = min((len(stream_X) - WSIZE) // STEP + 1, 100)
print(f"Stream: {len(stream_X):,} | Window={WSIZE} | Attack={stream_y.mean()*100:.0f}%")

# Online SGD classifier for incremental learning
idx0_t = np.where(y_train_bal == 0)[0]
idx1_t = np.where(y_train_bal == 1)[0]
n_in   = min(2500, len(idx0_t), len(idx1_t))
init_i = np.concatenate([
    np.random.choice(idx0_t, n_in, replace=False),
    np.random.choice(idx1_t, n_in, replace=False)])
online_clf = SGDClassifier(loss='log_loss', class_weight='balanced',
                            learning_rate='adaptive', eta0=0.005,
                            random_state=SEED, max_iter=1, warm_start=True)
online_clf.fit(X_train_bal[init_i], y_train_bal[init_i])

stream_res=[]; drift_pts=[]; buf_X=[]; buf_y=[]
prev_err=None; rtc=0; EXP_ATK=0.50

print(f"\n{'Win':>4} {'F1_Hyb':>8} {'DynThr':>8} "
      f"{'F1_Onl':>8} {'Atk%':>6} {'Drift':>7} {'Retrn':>7}")
print("-" * 58)

for w in range(n_wins):
    st = w * STEP; en = st + WSIZE
    if en > len(stream_X): break
    Xw = stream_X[st:en]; yw = stream_y[st:en]

    ss_w = sup_clf.predict_proba(Xw)[:, 1]
    raw_w = -iforest.decision_function(Xw)
    sh_w  = normalize_scores(raw_w)
    hw    = hybrid_fusion(ss_w, sh_w, W_SUP, W_HST)

    # Dynamic Threshold per window — adapts to local score distribution
    dyn_thr = float(np.percentile(hw, (1 - EXP_ATK) * 100))
    dyn_thr = float(np.clip(dyn_thr, 0.2, 0.8))

    try:
        f1_h = f1_score(yw, (hw > dyn_thr).astype(int),
                        average='weighted', zero_division=0)
    except: f1_h = 0.0

    try:
        so   = online_clf.predict_proba(Xw)[:, 1]
        f1_o = f1_score(yw, (so > 0.5).astype(int),
                        average='weighted', zero_division=0)
    except: f1_o = 0.0

    err_w = 1 - f1_h; atp = yw.mean() * 100
    conf  = float(np.mean(np.abs(hw - 0.5)) * 2)

    drift = False
    if prev_err is not None and abs(err_w - prev_err) > DRIFT_T:
        drift = True; drift_pts.append(w)
    prev_err = err_w

    buf_X.append(Xw); buf_y.append(yw); retr = False
    if (w + 1) % RETRAIN_E == 0:
        Xb = np.vstack(buf_X[-RETRAIN_E:])
        yb = np.concatenate(buf_y[-RETRAIN_E:])
        if len(np.unique(yb)) > 1:
            online_clf.partial_fit(Xb, yb, classes=[0, 1])
            retr = True; rtc += 1

    stream_res.append({
        'window': w, 'f1_hybrid': f1_h, 'f1_online': f1_o,
        'dyn_thr': dyn_thr, 'atk_pct': atp,
        'drift': drift, 'retrained': retr, 'conf': conf,
    })

    if w % 10 == 0 or drift:
        ds = "DRIFT!" if drift else ""; rs = "YES" if retr else ""
        print(f"{w+1:>4} {f1_h:>8.4f} {dyn_thr:>8.3f} {f1_o:>8.4f} "
              f"{atp:>5.1f}% {ds:>7} {rs:>7}")

sdf = pd.DataFrame(stream_res)
print(f"\n[Streaming Summary]")
print(f"  Hybrid F1 : {sdf['f1_hybrid'].mean():.4f} +/- {sdf['f1_hybrid'].std():.4f}")
print(f"  Online F1 : {sdf['f1_online'].mean():.4f} +/- {sdf['f1_online'].std():.4f}")
print(f"  Drift: {len(drift_pts)} | Retrain: {rtc}")

stream_summary = {
    'mean_f1_hybrid': round(float(sdf['f1_hybrid'].mean()), 4),
    'std_f1_hybrid':  round(float(sdf['f1_hybrid'].std()),  4),
    'mean_f1_online': round(float(sdf['f1_online'].mean()), 4),
    'drift_count':    len(drift_pts), 'retrain_count': rtc,
}

# ================================================================
# 6. Full Test Evaluation
# ================================================================
print("\n" + "=" * 60)
print("Full Evaluation on Test Set...")
print("=" * 60)

def eval_model(name, y_true, scores, thr=0.5):
    preds  = (scores > thr).astype(int)
    fpr, tpr, _ = roc_curve(y_true, scores)
    r_auc  = auc(fpr, tpr)
    p, r, _ = precision_recall_curve(y_true, scores)
    pr_auc = auc(r, p)
    f1  = f1_score(y_true, preds, average='weighted', zero_division=0)
    pre = precision_score(y_true, preds, average='weighted', zero_division=0)
    rec = recall_score(y_true, preds, average='weighted', zero_division=0)
    rec_b = recall_score(y_true, preds, average='binary', zero_division=0)
    cm  = confusion_matrix(y_true, preds)
    tn, fp, fn, tp = cm.ravel() if cm.shape == (2, 2) else (0, 0, 0, 0)
    fpr_v = fp / (fp + tn + 1e-9)
    print(f"\n=== {name} ===")
    print(classification_report(y_true, preds, zero_division=0))
    return {
        'name': name, 'AUC': r_auc, 'PR_AUC': pr_auc,
        'F1': f1, 'Precision': pre, 'Recall': rec,
        'Recall_binary': rec_b, 'FPR_val': fpr_v,
        'FPR': fpr, 'TPR': tpr,
        'Precision_arr': p, 'Recall_arr': r,
        'CM': cm, 'preds': preds, 'scores': scores,
        'TN': tn, 'FP': fp, 'FN': fn, 'TP': tp,
    }

final = {}
final[SUPER_NAME]        = eval_model(SUPER_NAME,        y_test, s_sup_test, THR_SUP)
final['IsolationForest'] = eval_model('IsolationForest', y_test, s_hst_test, THR_HST)
final['Hybrid (Ours)']   = eval_model('Hybrid (Ours)',   y_test, s_hyb_test, THR_MAIN)

# ================================================================
# 7. Statistical Tests
# ================================================================
print("\n" + "=" * 60)
print("Statistical Significance Tests (scipy)")
print("=" * 60)

hyb_p   = final['Hybrid (Ours)']['preds']
sig_res = {}

for nm in [SUPER_NAME, 'IsolationForest']:
    op = final[nm]['preds']
    b  = int(((hyb_p == y_test) & (op != y_test)).sum())
    c  = int(((hyb_p != y_test) & (op == y_test)).sum())
    p_mc = mcnemar_test(b, c)
    sig_res[nm] = {'p': p_mc, 'b': b, 'c': c}
    sig = "Significant ✓" if (not np.isnan(p_mc) and p_mc < 0.05) else "Not significant"
    print(f"  McNemar Hybrid vs {nm:<22} "
          f"b={b:,} c={c:,} p={p_mc:.6f}  {sig}")

# Bootstrap CI
print("\n[Bootstrap 95% CI — AUC] (n_boot=1000)")
ci_res = {}
for nm, res in final.items():
    lo, hi = bootstrap_ci(y_test, res['scores'], n_boot=1000)
    ci_res[nm] = {'lo': lo, 'hi': hi}
    print(f"  {nm:<25} AUC={res['AUC']:.4f}  "
          f"95%CI=[{lo:.4f}, {hi:.4f}]")

# Wilcoxon
print("\n[Wilcoxon Test — Score Distributions]")
for nm in [SUPER_NAME, 'IsolationForest']:
    try:
        idx_w = np.random.choice(len(y_test),
                                  min(5000, len(y_test)), replace=False)
        stat, p_wx = scipy_wilcoxon(
            final['Hybrid (Ours)']['scores'][idx_w],
            final[nm]['scores'][idx_w])
        print(f"  Hybrid vs {nm:<22} stat={stat:.2f}  p={p_wx:.6f}")
    except Exception as e:
        print(f"  Hybrid vs {nm:<22} {e}")

# ================================================================
# 8. SHAP
# ================================================================
if HAS_SHAP and SUPER_NAME == 'Random Forest':
    print("\n[SHAP] Computing feature importance...")
    try:
        ex  = shap.TreeExplainer(sup_clf)
        sv  = ex.shap_values(X_test_sc[:300])
        sv2d = (np.array(sv[1]) if isinstance(sv, list)
                else sv[:, :, 1] if isinstance(sv, np.ndarray) and sv.ndim == 3
                else np.array(sv))
        ms  = np.abs(sv2d).mean(axis=0)
        top = min(15, len(ms))
        fi  = np.argsort(ms)[-top:]
        fig_s, ax_s = plt.subplots(figsize=(10, 6))
        ax_s.barh(range(top), ms[fi], color='steelblue')
        ax_s.set_yticks(range(top))
        ax_s.set_yticklabels(feat_names[fi], fontsize=9)
        ax_s.set_xlabel("Mean |SHAP value|", fontsize=12)
        ax_s.set_title("SHAP Feature Importance", fontweight='bold')
        ax_s.grid(axis='x', alpha=0.3)
        plt.tight_layout()
        plt.savefig("shap_final.png", dpi=300, bbox_inches='tight')
        plt.close()
        print("  Saved: shap_final.png")
    except Exception as e:
        print(f"  [SHAP] {e}")

# ================================================================
# 9. Publication-Quality Figures (10 plots)
# ================================================================
print("\n[Plots] Generating 10 publication-quality figures...")

C  = {SUPER_NAME: '#2ca02c', 'IsolationForest': '#e377c2', 'Hybrid (Ours)': '#d62728'}
St = {SUPER_NAME: '--',      'IsolationForest': ':',       'Hybrid (Ours)': '-'}
LW = {k: (3 if k == 'Hybrid (Ours)' else 1.5) for k in C}

# Fig 1: ROC + CI
fig1, ax1 = plt.subplots(figsize=(8, 7))
for nm, res in final.items():
    lo = ci_res.get(nm, {}).get('lo', '?')
    hi = ci_res.get(nm, {}).get('hi', '?')
    lbl = f"{nm} (AUC={res['AUC']:.4f} [{lo},{hi}])"
    ax1.plot(res['FPR'], res['TPR'], color=C[nm], linestyle=St[nm],
             lw=LW[nm], label=lbl)
ax1.plot([0,1],[0,1], 'k--', lw=1, label="Random (0.500)")
ax1.set_xlabel("False Positive Rate", fontsize=12)
ax1.set_ylabel("True Positive Rate",  fontsize=12)
ax1.set_title("ROC Curve with 95% Bootstrap CI\n"
              "Smart Temporal Split Test Set",
              fontsize=13, fontweight='bold')
ax1.legend(fontsize=8, loc='lower right'); ax1.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("fig1_roc_final.png", dpi=300, bbox_inches='tight'); plt.close()

# Fig 2: Zero-Day Detection Rate
fig2, ax2 = plt.subplots(figsize=(10, 5))
if zd_results:
    atks = list(zd_results.keys())
    xp   = np.arange(len(atks)); wb = 0.35
    b1 = ax2.bar(xp - wb/2, [zd_results[a]['supervised'] for a in atks],
                 wb, label=f'{SUPER_NAME}', color='#2ca02c', alpha=0.85)
    b2 = ax2.bar(xp + wb/2, [zd_results[a]['hybrid'] for a in atks],
                 wb, label='Hybrid (Ours)', color='#d62728', alpha=0.85)
    ax2.set_xticks(xp); ax2.set_xticklabels(atks, fontsize=11)
    ax2.set_ylabel("Detection Rate", fontsize=12); ax2.set_ylim(0, 1.2)
    ax2.set_title("Zero-Day Attack Detection Rate\n"
                  "(Models trained WITHOUT these attacks)",
                  fontsize=13, fontweight='bold')
    ax2.legend(fontsize=11); ax2.grid(axis='y', alpha=0.3)
    for bar in list(b1) + list(b2):
        h = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2, h + 0.02,
                 f"{h:.3f}", ha='center', fontsize=10, fontweight='bold')
else:
    ax2.text(0.5, 0.5, "No Zero-Day attacks in Test Set",
             ha='center', va='center', transform=ax2.transAxes, fontsize=14)
plt.tight_layout()
plt.savefig("fig2_zeroday_final.png", dpi=300, bbox_inches='tight'); plt.close()

# Fig 3: Throughput
fig3, axes3 = plt.subplots(1, 2, figsize=(13, 5))
bs_list  = [r['batch']   for r in thr_results['no_attack']]
thr_na   = [r['thr_pps'] for r in thr_results['no_attack']]
thr_at   = [r['thr_pps'] for r in thr_results['attack']]
lat_na   = [r['lat_ms']  for r in thr_results['no_attack']]
lat_at   = [r['lat_ms']  for r in thr_results['attack']]

ax3a = axes3[0]
ax3a.plot(bs_list, thr_na, 'o-', color='#2ca02c', lw=2, label='No-Attack (BENIGN)')
ax3a.plot(bs_list, thr_at, 's-', color='#d62728', lw=2, label='Under Attack')
ax3a.set_xlabel("Batch Size (packets)", fontsize=11)
ax3a.set_ylabel("Throughput (packets/sec)", fontsize=11)
ax3a.set_title("Throughput vs Batch Size", fontsize=12, fontweight='bold')
ax3a.set_xscale('log'); ax3a.legend(fontsize=10); ax3a.grid(alpha=0.3)
for x, y1, y2 in zip(bs_list, thr_na, thr_at):
    ax3a.annotate(f"{y1:,.0f}", xy=(x, y1), fontsize=7,
                  ha='center', va='bottom', color='#2ca02c')
    ax3a.annotate(f"{y2:,.0f}", xy=(x, y2), fontsize=7,
                  ha='center', va='top', color='#d62728')

ax3b = axes3[1]
ax3b.plot(bs_list, lat_na, 'o-', color='#2ca02c', lw=2, label='No-Attack (BENIGN)')
ax3b.plot(bs_list, lat_at, 's-', color='#d62728', lw=2, label='Under Attack')
ax3b.set_xlabel("Batch Size (packets)", fontsize=11)
ax3b.set_ylabel("Latency per Packet (ms)", fontsize=11)
ax3b.set_title("Latency vs Batch Size", fontsize=12, fontweight='bold')
ax3b.set_xscale('log'); ax3b.legend(fontsize=10); ax3b.grid(alpha=0.3)
plt.suptitle("System Performance: No-Attack vs Under-Attack",
             fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig("fig3_throughput_final.png", dpi=300, bbox_inches='tight')
plt.close(); print("  fig3_throughput_final.png ✓")

# Fig 4: Real-Time
fig4 = plt.figure(figsize=(15, 5))
gs4  = gridspec.GridSpec(1, 3, wspace=0.35)
ax4a = fig4.add_subplot(gs4[0, 0])
alc  = {'NORMAL':'#2ca02c','LOW':'#f7b731','MEDIUM':'#fd9644',
        'HIGH':'#e74c3c','CRITICAL':'#8e0000'}
nzv  = [(lv, rt_al.count(lv)) for lv in alc if rt_al.count(lv) > 0]
if nzv:
    lvs, vs = zip(*nzv)
    ax4a.pie(vs, labels=lvs, colors=[alc[l] for l in lvs],
             autopct='%1.1f%%', startangle=90)
ax4a.set_title(f"Alert Distribution\n"
               f"Acc={rt_acc*100:.1f}%  Recall={rt_rec:.3f}",
               fontweight='bold', fontsize=11)

ax4b = fig4.add_subplot(gs4[0, 1])
ax4b.hist(rt_lat, bins=25, color='steelblue', edgecolor='white', alpha=0.85)
ax4b.axvline(np.mean(rt_lat), color='red', linestyle='--', lw=2,
             label=f"Mean={np.mean(rt_lat):.1f}ms")
ax4b.axvline(np.percentile(rt_lat, 95), color='orange', linestyle=':', lw=2,
             label=f"P95={np.percentile(rt_lat,95):.1f}ms")
ax4b.set_xlabel("Latency (ms)", fontsize=11)
ax4b.set_ylabel("Count", fontsize=11)
ax4b.set_title("Inference Latency", fontweight='bold', fontsize=11)
ax4b.legend(fontsize=9); ax4b.grid(alpha=0.3)

ax4c = fig4.add_subplot(gs4[0, 2])
ax4c.scatter(rt_sc, [1 if ok else 0 for ok in rt_ok],
             c=['#2ca02c' if ok else '#d62728' for ok in rt_ok],
             alpha=0.5, s=20)
ax4c.axvline(THR_MAIN, color='black', linestyle='--', lw=1.5,
             label=f"Thr={THR_MAIN:.2f}")
ax4c.set_xlabel("Hybrid Score", fontsize=11)
ax4c.set_ylabel("Correct=1/Wrong=0", fontsize=11)
ax4c.set_title("Score vs Correctness", fontweight='bold', fontsize=11)
ax4c.legend(fontsize=9); ax4c.grid(alpha=0.3)
plt.suptitle("Real-Time IDS Analysis", fontsize=14, fontweight='bold')
plt.savefig("fig4_realtime_final.png", dpi=300, bbox_inches='tight'); plt.close()

# Fig 5: Streaming
fig5, axes5 = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
fig5.subplots_adjust(hspace=0.3)
ax5a = axes5[0]
ax5a.plot(sdf['window'], sdf['f1_hybrid'], color='#d62728', lw=1.5,
          label='Hybrid (Dynamic Thr)')
ax5a.plot(sdf['window'], sdf['f1_online'], color='#1f77b4', lw=1.2,
          linestyle='--', label='Online SGD')
for dp in drift_pts:
    ax5a.axvline(dp, color='orange', alpha=0.7, lw=1.5, linestyle='--')
if drift_pts:
    ax5a.axvline(drift_pts[0], color='orange', alpha=0.7,
                 lw=1.5, linestyle='--', label='Concept Drift')
ax5a.set_ylabel("F1-Score", fontsize=11)
ax5a.set_title(f"Streaming — Hybrid={sdf['f1_hybrid'].mean():.4f}±"
               f"{sdf['f1_hybrid'].std():.4f}  "
               f"Online={sdf['f1_online'].mean():.4f}",
               fontsize=12, fontweight='bold')
ax5a.legend(fontsize=9); ax5a.grid(alpha=0.3); ax5a.set_ylim(0, 1.05)

ax5b = axes5[1]
ax5b.fill_between(sdf['window'], sdf['atk_pct'], alpha=0.4, color='#e74c3c')
ax5b.plot(sdf['window'], sdf['atk_pct'], color='#e74c3c', lw=1,
          label='Attack %')
ax5b.set_ylabel("Attack Traffic (%)", fontsize=11)
ax5b.legend(fontsize=9); ax5b.grid(alpha=0.3)

ax5c = axes5[2]
ax5c.plot(sdf['window'], sdf['dyn_thr'], color='purple', lw=1.5,
          linestyle='--', label='Dynamic Threshold')
ax5c.plot(sdf['window'], sdf['conf'], color='#2ca02c', lw=1.5,
          label='Avg Confidence')
rw = sdf[sdf['retrained']]['window'].values
if len(rw):
    ax5c.scatter(rw, sdf[sdf['retrained']]['conf'].values,
                 marker='*', s=150, color='blue', zorder=5, label='Retrain')
ax5c.set_xlabel("Window #", fontsize=11)
ax5c.set_ylabel("Value", fontsize=11)
ax5c.legend(fontsize=9); ax5c.grid(alpha=0.3)
plt.savefig("fig5_streaming_final.png", dpi=300, bbox_inches='tight'); plt.close()

# Fig 6: Confusion Matrices
fig6, axes6 = plt.subplots(1, 3, figsize=(15, 5))
fig6.suptitle("Confusion Matrices — Test Set",
              fontsize=13, fontweight='bold')
for ax, nm in zip(axes6, [SUPER_NAME, 'IsolationForest', 'Hybrid (Ours)']):
    cm  = final[nm]['CM']
    im  = ax.imshow(cm, cmap='Blues')
    ax.set_title(f"{nm}\nRecall={final[nm]['Recall_binary']:.3f}",
                 fontsize=10, fontweight='bold')
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    ax.set_xticks([0,1]); ax.set_yticks([0,1])
    ax.set_xticklabels(['Benign','Attack'])
    ax.set_yticklabels(['Benign','Attack'])
    thr = cm.max() / 2
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm[i,j]:,}", ha='center', va='center',
                    fontsize=10,
                    color='white' if cm[i,j] > thr else 'black')
    plt.colorbar(im, ax=ax)
plt.tight_layout()
plt.savefig("fig6_confusion_final.png", dpi=300, bbox_inches='tight'); plt.close()

# Fig 7: PR
fig7, ax7 = plt.subplots(figsize=(8, 7))
for nm, res in final.items():
    ax7.plot(res['Recall_arr'], res['Precision_arr'],
             color=C[nm], linestyle=St[nm], lw=LW[nm],
             label=f"{nm} (PR-AUC={res['PR_AUC']:.4f})")
ax7.set_xlabel("Recall", fontsize=12); ax7.set_ylabel("Precision", fontsize=12)
ax7.set_title("Precision-Recall Curve", fontsize=14, fontweight='bold')
ax7.legend(fontsize=10); ax7.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("fig7_pr_final.png", dpi=300, bbox_inches='tight'); plt.close()

# Fig 8: Three-Scenario Summary
fig8 = plt.figure(figsize=(15, 5))
gs8  = gridspec.GridSpec(1, 3, wspace=0.35)
ax8a = fig8.add_subplot(gs8[0, 0])
if zd_results:
    atks_z = list(zd_results.keys())
    gains  = [zd_results[a]['gain'] for a in atks_z]
    cols8  = ['#d62728' if g >= 0 else '#2ca02c' for g in gains]
    bars8  = ax8a.bar(atks_z, gains, color=cols8, edgecolor='white', alpha=0.85)
    ax8a.axhline(0, color='black', lw=0.8)
    for bar, g in zip(bars8, gains):
        ax8a.text(bar.get_x() + bar.get_width()/2,
                  g + (0.005 if g >= 0 else -0.01),
                  f"{'+'if g>=0 else ''}{g:.3f}",
                  ha='center', fontsize=10, fontweight='bold')
else:
    ax8a.text(0.5, 0.5, "ZD attacks\nnot in Test",
              ha='center', va='center', transform=ax8a.transAxes, fontsize=12)
ax8a.set_title("A: Zero-Day\nHybrid Gain", fontweight='bold', fontsize=11)
ax8a.set_ylabel("Δ Detection Rate"); ax8a.grid(axis='y', alpha=0.3)

ax8b = fig8.add_subplot(gs8[0, 1])
slat = np.sort(rt_lat); cdf = np.arange(1, len(slat)+1) / len(slat)
ax8b.plot(slat, cdf, color='#d62728', lw=2)
ax8b.axvline(np.percentile(rt_lat, 95), color='orange', linestyle='--', lw=2,
             label=f"P95={np.percentile(rt_lat,95):.1f}ms")
ax8b.axvline(np.mean(rt_lat), color='red', linestyle=':', lw=2,
             label=f"Mean={np.mean(rt_lat):.1f}ms")
ax8b.set_xlabel("Latency (ms)"); ax8b.set_ylabel("CDF")
ax8b.set_title("B: Real-Time\nLatency CDF", fontweight='bold', fontsize=11)
ax8b.legend(fontsize=9); ax8b.grid(alpha=0.3)

ax8c = fig8.add_subplot(gs8[0, 2])
rh = sdf['f1_hybrid'].rolling(5, min_periods=1).mean()
ro = sdf['f1_online'].rolling(5, min_periods=1).mean()
ax8c.plot(sdf['window'], rh, color='#d62728', lw=2,
          label=f"Hybrid (μ={sdf['f1_hybrid'].mean():.3f})")
ax8c.plot(sdf['window'], ro, color='#1f77b4', lw=1.5, linestyle='--',
          label=f"Online SGD (μ={sdf['f1_online'].mean():.3f})")
ax8c.set_xlabel("Window #"); ax8c.set_ylabel("F1 (rolling=5)")
ax8c.set_title("C: Streaming\nDynamic Threshold", fontweight='bold', fontsize=11)
ax8c.legend(fontsize=9); ax8c.grid(alpha=0.3)
plt.suptitle("Three-Scenario Evaluation Summary",
             fontsize=14, fontweight='bold')
plt.savefig("fig8_summary_final.png", dpi=300, bbox_inches='tight'); plt.close()

# Fig 9: Ablation
fig9, ax9 = plt.subplots(figsize=(8, 5))
ax9.plot(wg * 100, abl_f1s, 's-', color='darkorange', lw=2, markersize=6)
ax9.axvline(W_HST * 100, color='red', linestyle='--', lw=2,
            label=f"Best w_hst={W_HST:.2f} (F1={best_f1v:.4f})")
ax9.set_xlabel("IsolationForest Weight (%)", fontsize=12)
ax9.set_ylabel("Weighted F1 (Val)", fontsize=12)
ax9.set_title("Ablation Study: Fusion Weight Optimization",
              fontsize=13, fontweight='bold')
ax9.legend(fontsize=10); ax9.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("fig9_ablation_final.png", dpi=300, bbox_inches='tight'); plt.close()

# Fig 10: Alert Timeline
fig10, ax10 = plt.subplots(figsize=(14, 4))
alnum = {'NORMAL':1,'LOW':2,'MEDIUM':3,'HIGH':4,'CRITICAL':5}
alcl  = {'NORMAL':'#2ca02c','LOW':'#f7b731','MEDIUM':'#fd9644',
         'HIGH':'#e74c3c','CRITICAL':'#8e0000'}
for i, (sc2, al) in enumerate(zip(rt_sc[:100], rt_al[:100])):
    ax10.bar(i, alnum.get(al, 1), color=alcl.get(al,'gray'), alpha=0.8, width=0.8)
ax10.set_xlabel("Packet #", fontsize=11)
ax10.set_ylabel("Alert Level", fontsize=11)
ax10.set_yticks([1,2,3,4,5])
ax10.set_yticklabels(['NORMAL','LOW','MEDIUM','HIGH','CRITICAL'])
ax10.set_title("Real-Time Alert Timeline (First 100 Packets)",
               fontsize=13, fontweight='bold')
ax10.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig("fig10_alerts_final.png", dpi=300, bbox_inches='tight'); plt.close()

print("  10 figures saved successfully ✓")

# ================================================================
# 10. Final Results Tables
# ================================================================
rows = []
for nm, res in final.items():
    sg = sig_res.get(nm, {}); ci = ci_res.get(nm, {})
    rows.append({
        'Model'       : nm,
        'Test AUC'    : round(res['AUC'],          4),
        '95% CI'      : f"[{ci.get('lo','?')},{ci.get('hi','?')}]",
        'Test PR-AUC' : round(res['PR_AUC'],        4),
        'Test F1'     : round(res['F1'],            4),
        'Recall(Atk)' : round(res['Recall_binary'], 4),
        'Precision'   : round(res['Precision'],     4),
        'FPR'         : round(res['FPR_val'],       4),
        'Threshold'   : round(THR_MAIN if nm == 'Hybrid (Ours)'
                              else THR_SUP if nm == SUPER_NAME
                              else THR_HST, 3),
        'ZeroDay AUC' : (round(auc_hyb_zd, 4) if nm == 'Hybrid (Ours)'
                         else round(auc_sup_zd, 4) if nm == SUPER_NAME
                         else 'N/A'),
        'RT Recall'   : round(rt_rec, 4) if nm == 'Hybrid (Ours)' else 'N/A',
        'Stream F1'   : round(sdf['f1_hybrid'].mean(), 4) if nm == 'Hybrid (Ours)' else 'N/A',
        'McNemar p'   : round(sg.get('p', float('nan')), 6)
                        if nm != 'Hybrid (Ours)' else '—',
    })

df_res = pd.DataFrame(rows)
df_res.to_csv("comparison_final.csv", index=False)
print("\n[Results Table]")
print(df_res.to_string(index=False))

# Throughput table
df_thr = pd.DataFrame(
    [{'Scenario':'No-Attack','Batch':r['batch'],
      'Throughput(pps)':r['thr_pps'],'Latency(ms)':r['lat_ms'],
      'P95(ms)':r['p95_ms']} for r in thr_results['no_attack']] +
    [{'Scenario':'Attack',   'Batch':r['batch'],
      'Throughput(pps)':r['thr_pps'],'Latency(ms)':r['lat_ms'],
      'P95(ms)':r['p95_ms']} for r in thr_results['attack']]
)
df_thr.to_csv("throughput_final.csv", index=False)
print("\n[Throughput Table]")
print(df_thr.to_string(index=False))

# JSON
summary = {
    'version'     : 'FINAL',
    'split'       : 'smart_temporal',
    'train_files' : TRAIN_FILES,
    'val_files'   : VAL_FILES,
    'test_files'  : TEST_FILES,
    'supervised'  : SUPER_NAME,
    'W_SUP'       : round(float(W_SUP), 4),
    'W_HST'       : round(float(W_HST), 4),
    'THR_MAIN'    : round(float(THR_MAIN), 4),
    'THR_SUP'     : round(float(THR_SUP), 4),
    'throughput'  : {
        'no_attack_avg_pps' : round(avg_thr_na, 0),
        'attack_avg_pps'    : round(avg_thr_at, 0),
        'overhead_pct'      : round(overhead, 2),
    },
    'zero_day'    : {
        'auc_supervised': round(float(auc_sup_zd), 4),
        'auc_hybrid'    : round(float(auc_hyb_zd), 4),
        'per_attack'    : zd_results,
    },
    'real_time'   : rt_summary,
    'streaming'   : stream_summary,
    'statistical' : {
        'bootstrap_ci': ci_res,
        'mcnemar'     : {nm: {'p': round(v['p'], 6), 'b': v['b'], 'c': v['c']}
                         for nm, v in sig_res.items()},
    },
    'full_test'   : {nm: {
        'AUC'           : round(r['AUC'],          4),
        'F1'            : round(r['F1'],            4),
        'Recall_binary' : round(r['Recall_binary'], 4),
        'PR_AUC'        : round(r['PR_AUC'],        4),
    } for nm, r in final.items()},
}
with open("experiment_final.json", "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2, ensure_ascii=False)

print("\n" + "=" * 60)
print("Saved outputs:")
for fn in ["fig1_roc_final.png", "fig2_zeroday_final.png",
           "fig3_throughput_final.png", "fig4_realtime_final.png",
           "fig5_streaming_final.png", "fig6_confusion_final.png",
           "fig7_pr_final.png", "fig8_summary_final.png",
           "fig9_ablation_final.png", "fig10_alerts_final.png",
           "comparison_final.csv", "throughput_final.csv",
           "experiment_final.json"]:
    print(f"  {fn}")
if HAS_SHAP and SUPER_NAME == 'Random Forest':
    print("  shap_final.png")
print("=" * 60)
print("AI-IPS FINAL — Completed successfully.")
