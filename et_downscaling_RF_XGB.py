"""
=============================================================
 MODIS ET Downscaling — RF + XGBoost
 Datasets : Hyderabad (annual) | Miryalaguda Kharif | Rabi
 Author   : Your Name
=============================================================
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import warnings, joblib, os

from sklearn.ensemble        import RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.metrics         import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb

warnings.filterwarnings('ignore')
os.makedirs('outputs', exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

DATASETS = {
    'Hyderabad_Annual': {
        'file'    : 'Hyderabad_ET_training.csv',
        'features': ['Albedo', 'EVI', 'Elevation', 'LST',
                     'NDVI', 'NDWI', 'NIR_RED_ratio', 'SWIR_NIR_ratio', 'Slope'],
        'target'  : 'MODIS_ET',
        'color'   : '#1565C0',          # blue
    },
    'Miryalaguda_Kharif': {
        'file'    : 'Miryalaguda_ET_Kharif_training.csv',
        'features': ['Albedo', 'EVI', 'Elevation', 'LST', 'LSWI',
                     'NDVI', 'NDWI', 'NIR_RED_ratio', 'SWIR_NIR_ratio', 'Slope'],
        'target'  : 'MODIS_ET',
        'color'   : '#2E7D32',          # green
    },
    'Miryalaguda_Rabi': {
        'file'    : 'Miryalaguda_ET_Rabi_training.csv',
        'features': ['Albedo', 'EVI', 'Elevation', 'LST', 'LSWI',
                     'NDVI', 'NDWI', 'NIR_RED_ratio', 'SWIR_NIR_ratio', 'Slope'],
        'target'  : 'MODIS_ET',
        'color'   : '#E65100',          # orange
    },
}

RF_PARAMS = dict(
    n_estimators=200, max_depth=15, min_samples_leaf=2,
    max_features='sqrt', n_jobs=-1, random_state=42
)
XGB_PARAMS = dict(
    n_estimators=500, learning_rate=0.05, max_depth=6,
    subsample=0.8, colsample_bytree=0.8,
    reg_alpha=0.1, reg_lambda=1.0,
    eval_metric='rmse', early_stopping_rounds=30,
    random_state=42, verbosity=0
)

# ─────────────────────────────────────────────────────────────────────────────
# STEP 1 — LOAD & PREPROCESS
# ─────────────────────────────────────────────────────────────────────────────

def load_and_preprocess(cfg, name):
    print(f"\n{'='*60}")
    print(f"  DATASET : {name}")
    print(f"{'='*60}")

    df = pd.read_csv(cfg['file'])
    print(f"  Raw shape          : {df.shape}")
    print(f"  Columns            : {list(df.columns)}")

    # ── Drop metadata columns
    drop_cols = ['system:index', '.geo']
    df.drop(columns=[c for c in drop_cols if c in df.columns], inplace=True)

    # ── Missing values before cleaning
    mv = df.isnull().sum()
    if mv.sum() > 0:
        print(f"\n  Missing values:\n{mv[mv>0]}")
    else:
        print("  Missing values     : None")

    # ── Fix LST fill values (Landsat cloud fill = ~-124°C)
    #    Any LST below -10°C for India is physically impossible → set NaN
    lst_bad = (df['LST'] < -10).sum()
    if lst_bad > 0:
        print(f"  LST fill values    : {lst_bad:,} pixels with LST < -10°C → set to NaN")
        df.loc[df['LST'] < -10, 'LST'] = np.nan

    # ── Drop rows with NaN (after LST fix)
    before = len(df)
    df.dropna(subset=cfg['features'] + [cfg['target']], inplace=True)
    dropped = before - len(df)
    if dropped > 0:
        print(f"  Rows dropped       : {dropped:,} (NaN after LST fix)")

    # ── Clip NIR/RED ratio outliers (division by near-zero RED over water)
    p99 = df['NIR_RED_ratio'].quantile(0.99)
    clipped = (df['NIR_RED_ratio'] > p99).sum()
    df['NIR_RED_ratio'] = df['NIR_RED_ratio'].clip(upper=p99)
    if clipped > 0:
        print(f"  NIR_RED outliers   : {clipped:,} clipped at 99th pct ({p99:.2f})")

    # ── Drop duplicate Elevation column if present (keep lowercase 'elevation')
    if 'elevation' in df.columns and 'Elevation' in df.columns:
        df.drop(columns=['elevation'], inplace=True)
        print("  Elevation          : dropped duplicate lowercase 'elevation'")

    print(f"  Clean shape        : {df.shape}")

    # ── Basic stats
    print(f"\n  Target ({cfg['target']}) stats:")
    print(f"    mean={df[cfg['target']].mean():.2f}  std={df[cfg['target']].std():.2f}"
          f"  min={df[cfg['target']].min():.2f}  max={df[cfg['target']].max():.2f}")
    print(f"\n  LST stats (post-fix):")
    print(f"    mean={df['LST'].mean():.2f}  std={df['LST'].std():.2f}"
          f"  min={df['LST'].min():.2f}  max={df['LST'].max():.2f}")

    return df

# ─────────────────────────────────────────────────────────────────────────────
# STEP 2 — TRAIN / TEST SPLIT
# ─────────────────────────────────────────────────────────────────────────────

def split_data(df, cfg):
    X = df[cfg['features']]
    y = df[cfg['target']]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print(f"\n  Train: {len(X_train):,}  |  Test: {len(X_test):,}")
    return X, y, X_train, X_test, y_train, y_test

# ─────────────────────────────────────────────────────────────────────────────
# STEP 3 — TRAIN MODELS
# ─────────────────────────────────────────────────────────────────────────────

def train_rf(X_train, y_train, X_test, y_test, X, y):
    print("\n  --- Random Forest ---")
    model = RandomForestRegressor(**RF_PARAMS)
    model.fit(X_train, y_train)
    pred  = model.predict(X_test)

    rmse = np.sqrt(mean_squared_error(y_test, pred))
    mae  = mean_absolute_error(y_test, pred)
    r2   = r2_score(y_test, pred)

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    cv = cross_val_score(model, X, y, cv=kf, scoring='r2')

    print(f"    RMSE   : {rmse:.4f}")
    print(f"    MAE    : {mae:.4f}")
    print(f"    R²     : {r2:.4f}")
    print(f"    CV R²  : {cv.mean():.4f} ± {cv.std():.4f}")

    return model, pred, {'RMSE': rmse, 'MAE': mae, 'R2': r2,
                         'CV_R2': cv.mean(), 'CV_std': cv.std(), 'cv_scores': cv}

def train_xgb(X_train, y_train, X_test, y_test, X, y):
    print("\n  --- XGBoost ---")
    model = xgb.XGBRegressor(**XGB_PARAMS)
    model.fit(X_train, y_train,
              eval_set=[(X_test, y_test)], verbose=False)
    pred  = model.predict(X_test)

    rmse = np.sqrt(mean_squared_error(y_test, pred))
    mae  = mean_absolute_error(y_test, pred)
    r2   = r2_score(y_test, pred)

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    cv = cross_val_score(
        xgb.XGBRegressor(n_estimators=200, learning_rate=0.05,
                         max_depth=6, random_state=42, verbosity=0),
        X, y, cv=kf, scoring='r2'
    )

    print(f"    RMSE   : {rmse:.4f}")
    print(f"    MAE    : {mae:.4f}")
    print(f"    R²     : {r2:.4f}")
    print(f"    CV R²  : {cv.mean():.4f} ± {cv.std():.4f}")

    return model, pred, {'RMSE': rmse, 'MAE': mae, 'R2': r2,
                         'CV_R2': cv.mean(), 'CV_std': cv.std(), 'cv_scores': cv}

# ─────────────────────────────────────────────────────────────────────────────
# STEP 4 — PER-DATASET PLOTS
# ─────────────────────────────────────────────────────────────────────────────

def plot_dataset(name, cfg, features,
                 y_test, rf_pred, xgb_pred,
                 rf_metrics, xgb_metrics,
                 rf_model, xgb_model):

    c = cfg['color']
    fig = plt.figure(figsize=(20, 16))
    fig.suptitle(f'ET Downscaling — {name.replace("_"," ")}\nRandom Forest vs XGBoost',
                 fontsize=15, fontweight='bold', y=1.01)

    gs  = gridspec.GridSpec(3, 3, figure=fig, hspace=0.45, wspace=0.35)

    # ── 1. RF Actual vs Predicted
    ax = fig.add_subplot(gs[0, 0])
    lims = [min(y_test.min(), rf_pred.min()), max(y_test.max(), rf_pred.max())]
    ax.scatter(y_test, rf_pred, alpha=0.35, s=12, color='steelblue')
    ax.plot(lims, lims, 'r--', lw=1.8)
    ax.set_xlabel('Actual ET (MODIS)')
    ax.set_ylabel('Predicted ET')
    ax.set_title(f'RF — Actual vs Predicted\nR²={rf_metrics["R2"]:.3f}  RMSE={rf_metrics["RMSE"]:.3f}')
    ax.grid(True, alpha=0.2)

    # ── 2. XGB Actual vs Predicted
    ax = fig.add_subplot(gs[0, 1])
    ax.scatter(y_test, xgb_pred, alpha=0.35, s=12, color='darkorange')
    ax.plot(lims, lims, 'r--', lw=1.8)
    ax.set_xlabel('Actual ET (MODIS)')
    ax.set_ylabel('Predicted ET')
    ax.set_title(f'XGB — Actual vs Predicted\nR²={xgb_metrics["R2"]:.3f}  RMSE={xgb_metrics["RMSE"]:.3f}')
    ax.grid(True, alpha=0.2)

    # ── 3. Metrics bar comparison
    ax = fig.add_subplot(gs[0, 2])
    metrics_names = ['RMSE', 'MAE', 'R²']
    rf_v  = [rf_metrics['RMSE'],  rf_metrics['MAE'],  rf_metrics['R2']]
    xgb_v = [xgb_metrics['RMSE'], xgb_metrics['MAE'], xgb_metrics['R2']]
    x_pos = np.arange(3)
    b1 = ax.bar(x_pos - 0.2, rf_v,  0.35, label='RF',  color='steelblue')
    b2 = ax.bar(x_pos + 0.2, xgb_v, 0.35, label='XGB', color='darkorange')
    ax.set_xticks(x_pos); ax.set_xticklabels(metrics_names)
    ax.set_title('Metrics Comparison'); ax.legend()
    for b in list(b1) + list(b2):
        ax.text(b.get_x() + b.get_width()/2, b.get_height() + 0.005,
                f'{b.get_height():.3f}', ha='center', va='bottom', fontsize=7.5)

    # ── 4. RF Residuals scatter
    ax = fig.add_subplot(gs[1, 0])
    rf_res = y_test - rf_pred
    ax.scatter(rf_pred, rf_res, alpha=0.35, s=12, color='steelblue')
    ax.axhline(0, color='red', linestyle='--', lw=1.8)
    ax.set_xlabel('Predicted ET'); ax.set_ylabel('Residual (Actual − Pred)')
    ax.set_title('RF — Residuals vs Predicted')
    ax.grid(True, alpha=0.2)

    # ── 5. XGB Residuals scatter
    ax = fig.add_subplot(gs[1, 1])
    xgb_res = y_test - xgb_pred
    ax.scatter(xgb_pred, xgb_res, alpha=0.35, s=12, color='darkorange')
    ax.axhline(0, color='red', linestyle='--', lw=1.8)
    ax.set_xlabel('Predicted ET'); ax.set_ylabel('Residual (Actual − Pred)')
    ax.set_title('XGB — Residuals vs Predicted')
    ax.grid(True, alpha=0.2)

    # ── 6. CV boxplot
    ax = fig.add_subplot(gs[1, 2])
    bp = ax.boxplot([rf_metrics['cv_scores'], xgb_metrics['cv_scores']],
                    labels=['RF', 'XGBoost'], patch_artist=True, widths=0.5)
    bp['boxes'][0].set_facecolor('steelblue')
    bp['boxes'][1].set_facecolor('darkorange')
    ax.set_ylabel('R² Score')
    ax.set_title('5-Fold Cross-Validation R²')
    ax.grid(True, alpha=0.2)
    # Add mean labels
    for i, scores in enumerate([rf_metrics['cv_scores'], xgb_metrics['cv_scores']]):
        ax.text(i+1, scores.mean(), f'μ={scores.mean():.3f}',
                ha='center', va='bottom', fontsize=8, fontweight='bold')

    # ── 7. RF Feature Importance
    ax = fig.add_subplot(gs[2, 0])
    rf_imp = pd.Series(rf_model.feature_importances_, index=features).sort_values()
    colors = ['#E53935' if rf_imp.iloc[-1] == v else 'steelblue' for v in rf_imp.values]
    rf_imp.plot(kind='barh', ax=ax, color=colors)
    ax.set_title('RF — Feature Importance')
    ax.set_xlabel('MDI Score')
    for i, v in enumerate(rf_imp.values):
        ax.text(v + 0.001, i, f'{v:.3f}', va='center', fontsize=8)

    # ── 8. XGB Feature Importance
    ax = fig.add_subplot(gs[2, 1])
    xgb_imp = pd.Series(xgb_model.feature_importances_, index=features).sort_values()
    colors = ['#E53935' if xgb_imp.iloc[-1] == v else 'darkorange' for v in xgb_imp.values]
    xgb_imp.plot(kind='barh', ax=ax, color=colors)
    ax.set_title('XGB — Feature Importance')
    ax.set_xlabel('F-Score')
    for i, v in enumerate(xgb_imp.values):
        ax.text(v + 0.001, i, f'{v:.3f}', va='center', fontsize=8)

    # ── 9. ET Distribution
    ax = fig.add_subplot(gs[2, 2])
    ax.hist(y_test,    bins=40, alpha=0.5, label='Actual ET',    color='grey')
    ax.hist(rf_pred,   bins=40, alpha=0.5, label='RF Predicted', color='steelblue')
    ax.hist(xgb_pred,  bins=40, alpha=0.5, label='XGB Predicted',color='darkorange')
    ax.set_xlabel('ET (mm/8-day)'); ax.set_ylabel('Frequency')
    ax.set_title('ET Distribution: Actual vs Predicted')
    ax.legend(fontsize=8)

    plt.savefig(f'outputs/{name}_evaluation.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n  Plot saved → outputs/{name}_evaluation.png")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 5 — RESIDUAL DETAIL PLOT (separate file, cleaner)
# ─────────────────────────────────────────────────────────────────────────────

def plot_residuals(name, y_test, rf_pred, xgb_pred, rf_metrics, xgb_metrics):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(f'Residual Analysis — {name.replace("_"," ")} | RF vs XGBoost',
                 fontsize=13, fontweight='bold')

    rf_res  = y_test.values - rf_pred
    xgb_res = y_test.values - xgb_pred

    # Panel 1: Scatter
    ax = axes[0]
    ax.scatter(rf_pred,  rf_res,  alpha=0.3, s=10, color='steelblue',  label='RF')
    ax.scatter(xgb_pred, xgb_res, alpha=0.3, s=10, color='darkorange', label='XGB')
    ax.axhline(0, color='red', linestyle='--', lw=1.8)
    ax.set_xlabel('Predicted ET'); ax.set_ylabel('Residual')
    ax.set_title('Residuals vs Predicted'); ax.legend(); ax.grid(True, alpha=0.3)

    # Panel 2: Histogram
    ax = axes[1]
    ax.hist(rf_res,  bins=50, alpha=0.6, color='steelblue',
            label=f'RF  (std={rf_res.std():.3f})')
    ax.hist(xgb_res, bins=50, alpha=0.6, color='darkorange',
            label=f'XGB (std={xgb_res.std():.3f})')
    ax.axvline(0, color='red', linestyle='--', lw=1.8)
    ax.set_xlabel('Residual'); ax.set_ylabel('Frequency')
    ax.set_title('Residual Distribution'); ax.legend(); ax.grid(True, alpha=0.3)

    # Panel 3: CDF
    ax = axes[2]
    rf_abs  = np.sort(np.abs(rf_res))
    xgb_abs = np.sort(np.abs(xgb_res))
    cdf = np.linspace(0, 1, len(rf_abs))
    ax.plot(rf_abs,  cdf, color='steelblue',  lw=2, label='RF')
    ax.plot(xgb_abs, cdf, color='darkorange', lw=2, label='XGB')
    ax.set_xlabel('Absolute Error'); ax.set_ylabel('Cumulative Proportion')
    ax.set_title('Cumulative Absolute Error'); ax.legend(); ax.grid(True, alpha=0.3)
    # Mark 90th percentile
    p90 = np.percentile(rf_abs, 90)
    ax.axvline(p90, color='grey', linestyle=':', lw=1.5)
    ax.text(p90 + 0.02, 0.5, f'90%ile\n{p90:.2f}', fontsize=8, color='grey')

    plt.tight_layout()
    plt.savefig(f'outputs/{name}_residuals.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Residuals saved    → outputs/{name}_residuals.png")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 6 — CROSS-DATASET SUMMARY PLOT
# ─────────────────────────────────────────────────────────────────────────────

def plot_cross_dataset_summary(all_results):
    """Bar chart comparing RF vs XGB across all 3 datasets for each metric."""
    datasets = list(all_results.keys())
    n = len(datasets)
    metrics_to_plot = ['RMSE', 'MAE', 'R2', 'CV_R2']
    metric_labels   = ['RMSE (mm/8-day)', 'MAE (mm/8-day)', 'R²', 'CV R²']

    fig, axes = plt.subplots(1, 4, figsize=(22, 5))
    fig.suptitle('Model Comparison Across All Datasets — RF vs XGBoost',
                 fontsize=13, fontweight='bold')

    x = np.arange(n)
    w = 0.35

    for ax, metric, label in zip(axes, metrics_to_plot, metric_labels):
        rf_vals  = [all_results[d]['RF'][metric]  for d in datasets]
        xgb_vals = [all_results[d]['XGB'][metric] for d in datasets]

        b1 = ax.bar(x - w/2, rf_vals,  w, label='RF',  color='steelblue',  alpha=0.85)
        b2 = ax.bar(x + w/2, xgb_vals, w, label='XGB', color='darkorange', alpha=0.85)

        ax.set_xticks(x)
        ax.set_xticklabels([d.replace('_', '\n') for d in datasets], fontsize=8)
        ax.set_title(label); ax.legend(fontsize=8); ax.grid(True, alpha=0.2, axis='y')

        for b in list(b1) + list(b2):
            ax.text(b.get_x() + b.get_width()/2, b.get_height() + 0.003,
                    f'{b.get_height():.3f}', ha='center', va='bottom', fontsize=7.5)

    plt.tight_layout()
    plt.savefig('outputs/cross_dataset_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("\n  Cross-dataset comparison → outputs/cross_dataset_comparison.png")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 7 — CROSS-DATASET FEATURE IMPORTANCE COMPARISON
# ─────────────────────────────────────────────────────────────────────────────

def plot_feature_importance_cross(all_models):
    """
    For each model (RF, XGB) show importances side by side for all datasets.
    Handles the fact that Miryalaguda has LSWI but Hyderabad does not.
    """
    fig, axes = plt.subplots(1, 2, figsize=(20, 7))
    fig.suptitle('Feature Importance Across Datasets', fontsize=13, fontweight='bold')

    colors_ds = {'Hyderabad_Annual': '#1565C0',
                 'Miryalaguda_Kharif': '#2E7D32',
                 'Miryalaguda_Rabi': '#E65100'}

    for ax, model_key, title in zip(axes, ['RF', 'XGB'],
                                    ['Random Forest (MDI)', 'XGBoost (F-Score)']):
        all_features = set()
        for ds, m in all_models.items():
            all_features.update(m[model_key]['features'])
        all_features = sorted(all_features)

        bar_w = 0.25
        x = np.arange(len(all_features))
        datasets_list = list(all_models.keys())

        for i, ds in enumerate(datasets_list):
            imp_dict = dict(zip(all_models[ds][model_key]['features'],
                                all_models[ds][model_key]['model'].feature_importances_))
            vals = [imp_dict.get(f, 0) for f in all_features]
            offset = (i - 1) * bar_w
            ax.bar(x + offset, vals, bar_w, label=ds.replace('_', ' '),
                   color=colors_ds[ds], alpha=0.85)

        ax.set_xticks(x)
        ax.set_xticklabels(all_features, rotation=35, ha='right', fontsize=9)
        ax.set_ylabel('Importance Score')
        ax.set_title(title)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.2, axis='y')

    plt.tight_layout()
    plt.savefig('outputs/feature_importance_cross_dataset.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  Feature importance cross → outputs/feature_importance_cross_dataset.png")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 8 — PRINT FINAL SUMMARY TABLE
# ─────────────────────────────────────────────────────────────────────────────

def print_summary(all_results):
    print("\n" + "="*70)
    print("  FINAL SUMMARY TABLE")
    print("="*70)
    rows = []
    for ds, res in all_results.items():
        for model_name, m in res.items():
            rows.append({
                'Dataset'   : ds.replace('_', ' '),
                'Model'     : model_name,
                'RMSE'      : round(m['RMSE'],  4),
                'MAE'       : round(m['MAE'],   4),
                'R²'        : round(m['R2'],    4),
                'CV R²'     : round(m['CV_R2'], 4),
                'CV std'    : round(m['CV_std'],4),
            })
    summary_df = pd.DataFrame(rows)
    print(summary_df.to_string(index=False))
    summary_df.to_csv('outputs/summary_results.csv', index=False)
    print("\n  Full table saved → outputs/summary_results.csv")
    return summary_df

# ─────────────────────────────────────────────────────────────────────────────
# MAIN PIPELINE
# ─────────────────────────────────────────────────────────────────────────────

def run_pipeline():
    all_results = {}
    all_models  = {}

    for name, cfg in DATASETS.items():

        # 1. Load
        df = load_and_preprocess(cfg, name)

        # 2. Split
        X, y, X_train, X_test, y_train, y_test = split_data(df, cfg)

        # 3. Train
        rf_model,  rf_pred,  rf_metrics  = train_rf( X_train, y_train, X_test, y_test, X, y)
        xgb_model, xgb_pred, xgb_metrics = train_xgb(X_train, y_train, X_test, y_test, X, y)

        # 4. Per-dataset plots
        plot_dataset(name, cfg, cfg['features'],
                     y_test, rf_pred, xgb_pred,
                     rf_metrics, xgb_metrics,
                     rf_model, xgb_model)
        plot_residuals(name, y_test, rf_pred, xgb_pred, rf_metrics, xgb_metrics)

        # 5. Save models
        joblib.dump(rf_model,  f'outputs/{name}_rf_model.pkl')
        xgb_model.save_model(  f'outputs/{name}_xgb_model.json')
        print(f"  Models saved       → outputs/{name}_*.pkl / *.json")

        # 6. Collect results
        all_results[name] = {'RF': rf_metrics, 'XGB': xgb_metrics}
        all_models[name]  = {
            'RF':  {'model': rf_model,  'features': cfg['features']},
            'XGB': {'model': xgb_model, 'features': cfg['features']},
        }

    # 7. Cross-dataset comparison plots
    print("\n" + "="*60)
    print("  CROSS-DATASET PLOTS")
    print("="*60)
    plot_cross_dataset_summary(all_results)
    plot_feature_importance_cross(all_models)

    # 8. Summary table
    summary = print_summary(all_results)

    print("\n" + "="*60)
    print("  ALL DONE")
    print("="*60)
    print("  Outputs saved in:  ./outputs/")
    print("  Files generated:")
    for f in sorted(os.listdir('outputs')):
        print(f"    {f}")

run_pipeline()