#!/usr/bin/env python3
from __future__ import annotations

import argparse, hashlib, json, math, platform, sys, warnings
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy, sklearn, statsmodels
from statsmodels.tsa.stattools import acf, adfuller, kpss, pacf

CAMPAIGN_ID = 'UGR16-STATISTICAL-MODELS-001'
DIAGNOSTIC_ID = 'UGR16-STATISTICAL-MARCH-DIAGNOSTICS-001'
EXPECTED_PROTOCOL_SHA256 = '031138a75340ed45aab68b17cec8b4ef07f49c7445554fa4cbe4abc5414b0699'
EXPECTED_MARCH_SHA256 = 'fd6daea6007f1b411617a5c67a489fa33c2efda35fdd13616cc66a40134fa1b1'
TARGET = 'bitrate_bps'
VAR_COLS = ['bitrate_bps', 'packet_rate_pps', 'flow_rate_fps']
TRAIN_FRAC = 0.70
ALPHA = 0.05
ACF_NLAGS = 288
PACF_NLAGS = 48
PACF_METHOD = 'ywadjusted'
CROSS_MAX_LAG = 48

@dataclass
class TestResult:
    variable: str
    representation: str
    nobs: int
    adf_statistic: float
    adf_pvalue: float
    adf_used_lag: int
    adf_nobs: int
    adf_icbest: float | None
    kpss_statistic: float
    kpss_pvalue: float
    kpss_lags: int
    adf_supports_stationarity: bool
    kpss_supports_stationarity: bool
    joint_classification: str
    warnings: str


def args_parser():
    p = argparse.ArgumentParser(description='Diagnóstico March según TFM-STAT-PROTOCOL-001 FROZEN.')
    p.add_argument('--input', type=Path, default=Path('data/processed/ugr16/march_week3_prepared_5min.parquet'))
    p.add_argument('--protocol-file', type=Path, default=Path('docs/project_governance/003_statistical_models_protocol_2026-08-14.md'))
    p.add_argument('--metrics-dir', type=Path, default=Path('results/metrics'))
    p.add_argument('--figures-dir', type=Path, default=Path('results/figures/ugr16_statistical_march_diagnostics'))
    p.add_argument('--prefix', default='ugr16_statistical_march_diagnostics')
    p.add_argument('--dpi', type=int, default=180)
    p.add_argument('--overwrite', action='store_true')
    return p.parse_args()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def require_hash(path: Path, expected: str, label: str) -> str:
    if not path.is_file():
        raise FileNotFoundError(f'No existe {label}: {path}')
    observed = sha256_file(path)
    if observed != expected:
        raise RuntimeError(f'SHA-256 inválido para {label}. Esperado={expected} Observado={observed}')
    return observed


def reserve(path: Path, overwrite: bool):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not overwrite:
        raise FileExistsError(f'Ya existe: {path}. Usa --overwrite para sobrescribir.')


def load_validate(path: Path) -> tuple[pd.DataFrame, str]:
    h = require_hash(path, EXPECTED_MARCH_SHA256, 'March 5min')
    df = pd.read_parquet(path)
    if 'timestamp' not in df.columns:
        if df.index.name == 'timestamp':
            df = df.reset_index()
        else:
            raise ValueError('Falta timestamp.')
    df['timestamp'] = pd.to_datetime(df['timestamp'], errors='raise')
    df = df.sort_values('timestamp').reset_index(drop=True)
    missing = [c for c in ['timestamp', *VAR_COLS] if c not in df.columns]
    if missing: raise ValueError(f'Faltan columnas: {missing}')
    if df['timestamp'].duplicated().any(): raise ValueError('Timestamps duplicados.')
    vals = df[VAR_COLS].to_numpy(dtype=float)
    if not np.isfinite(vals).all(): raise ValueError('NaN/Inf en variables obligatorias.')
    if (df[VAR_COLS] < 0).any().any(): raise ValueError('Valores negativos no válidos.')
    d = df['timestamp'].diff().dropna()
    if not (d == pd.Timedelta(minutes=5)).all(): raise ValueError('La serie no es continua cada 5 minutos.')
    return df, h


def split(df: pd.DataFrame):
    n = int(len(df) * TRAIN_FRAC)
    train, val = df.iloc[:n].copy(), df.iloc[n:].copy()
    if train['timestamp'].max() >= val['timestamp'].min(): raise RuntimeError('Cronología inválida.')
    return train, val


def classify(adf_p: float, kpss_p: float):
    a = bool(adf_p < ALPHA)
    k = bool(kpss_p >= ALPHA)
    if a and k: joint = 'stationary'
    elif (not a) and (not k): joint = 'nonstationary'
    else: joint = 'ambiguous'
    return a, k, joint


def stationarity_one(s: pd.Series, variable: str, representation: str) -> TestResult:
    x = pd.Series(s, dtype='float64').dropna()
    if len(x) < 20 or math.isclose(float(x.std(ddof=0)), 0.0):
        raise ValueError(f'Serie no apta: {variable}/{representation}')
    caught = []
    with warnings.catch_warnings(record=True) as ws:
        warnings.simplefilter('always')
        adf_r = adfuller(x.to_numpy(), regression='c', autolag='AIC')
        kpss_r = kpss(x.to_numpy(), regression='c', nlags='auto')
        caught = [f'{w.category.__name__}: {w.message}' for w in ws]
    adf_stat, adf_p, adf_lag, adf_nobs, _, adf_icbest = adf_r
    kpss_stat, kpss_p, kpss_lags, _ = kpss_r
    a, k, joint = classify(float(adf_p), float(kpss_p))
    return TestResult(variable, representation, len(x), float(adf_stat), float(adf_p), int(adf_lag), int(adf_nobs), float(adf_icbest) if adf_icbest is not None else None, float(kpss_stat), float(kpss_p), int(kpss_lags), a, k, joint, ' | '.join(caught))


def stationarity_table(train: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for c in VAR_COLS:
        rows.append(asdict(stationarity_one(train[c], c, 'level')))
        rows.append(asdict(stationarity_one(train[c].diff(), c, 'first_difference')))
    return pd.DataFrame(rows)


def decisions(st: pd.DataFrame) -> dict:
    target = st[(st.variable == TARGET) & (st.representation == 'level')].iloc[0]
    tc = str(target.joint_classification)
    d = [0] if tc == 'stationary' else ([1] if tc == 'nonstationary' else [0, 1])
    levels = st[(st.variable.isin(VAR_COLS)) & (st.representation == 'level')]
    cls = dict(zip(levels.variable.astype(str), levels.joint_classification.astype(str)))
    var_repr = 'levels' if all(v == 'stationary' for v in cls.values()) else 'first_differences'
    return {'stationarity_alpha': ALPHA, 'target_level_classification': tc, 'arima_d_candidates_from_diagnostic': d, 'var_level_classifications': cls, 'var_representation_from_protocol': var_repr}


def acf_table(train: pd.DataFrame):
    y, ci = acf(train[TARGET].to_numpy(float), nlags=ACF_NLAGS, alpha=0.05, fft=True, missing='raise')
    l = np.arange(len(y))
    return pd.DataFrame({'lag_steps': l, 'lag_minutes': l * 5, 'acf': y, 'ci_low': ci[:,0], 'ci_high': ci[:,1]})


def pacf_table(train: pd.DataFrame):
    y, ci = pacf(train[TARGET].to_numpy(float), nlags=PACF_NLAGS, method=PACF_METHOD, alpha=0.05)
    l = np.arange(len(y))
    return pd.DataFrame({'lag_steps': l, 'lag_minutes': l * 5, 'pacf': y, 'ci_low': ci[:,0], 'ci_high': ci[:,1]})


def daily_profile(train: pd.DataFrame):
    x = train[['timestamp', TARGET]].copy(); x['hour'] = x.timestamp.dt.hour
    return x.groupby('hour', as_index=False).agg(mean_bitrate_bps=(TARGET,'mean'), median_bitrate_bps=(TARGET,'median'), std_bitrate_bps=(TARGET,'std'), n_intervals=(TARGET,'size'))


def var_frame(train: pd.DataFrame, repr_: str):
    x = train[VAR_COLS].astype(float).copy()
    return x if repr_ == 'levels' else x.diff().dropna()


def corr_table(x: pd.DataFrame):
    m = x[VAR_COLS].corr(method='pearson'); m.index.name = 'variable'; return m.reset_index()


def cross_corr(x: pd.DataFrame):
    rows=[]; target=x[TARGET].reset_index(drop=True)
    for src in ['packet_rate_pps','flow_rate_fps']:
        s=x[src].reset_index(drop=True)
        for lag in range(CROSS_MAX_LAG+1):
            xx, yy = (s, target) if lag == 0 else (s.iloc[:-lag].reset_index(drop=True), target.iloc[lag:].reset_index(drop=True))
            rows.append({'source':src,'target':TARGET,'lag_steps':lag,'lag_minutes':lag*5,'definition':'corr(source_t,target_t_plus_lag)','correlation':float(xx.corr(yy)),'n_pairs':len(xx)})
    return pd.DataFrame(rows)


def save_fig(base: Path, dpi: int):
    png, pdf = base.with_suffix('.png'), base.with_suffix('.pdf')
    plt.tight_layout(); plt.savefig(png, dpi=dpi, bbox_inches='tight'); plt.savefig(pdf, bbox_inches='tight'); plt.close()
    return [png,pdf]


def plot_series(train, base, dpi):
    plt.figure(figsize=(13,5.5)); plt.plot(train.timestamp, train[TARGET]/1e6, linewidth=1.0); plt.xlabel('Tiempo'); plt.ylabel('Bitrate (Mbit/s)'); plt.title("UGR'16 March — training inicial (70 %) — bitrate"); plt.grid(True, alpha=.25); plt.gcf().autofmt_xdate(); return save_fig(base,dpi)

def plot_diff(train, base, dpi):
    plt.figure(figsize=(13,5.5)); plt.plot(train.timestamp, train[TARGET].diff()/1e6, linewidth=1.0); plt.xlabel('Tiempo'); plt.ylabel('Δ bitrate (Mbit/s por intervalo)'); plt.title("UGR'16 March — primera diferencia — training"); plt.grid(True, alpha=.25); plt.gcf().autofmt_xdate(); return save_fig(base,dpi)

def plot_corr_lag(tab, field, title, base, dpi):
    q=tab[tab.lag_steps>0]; plt.figure(figsize=(13,5.5)); plt.plot(q.lag_steps, q[field], linewidth=1.0); plt.axhline(0, linewidth=.8); plt.xlabel('Lag (intervalos de 5 min)'); plt.ylabel(field.upper()); plt.title(title); plt.grid(True, alpha=.25); return save_fig(base,dpi)

def plot_daily(tab, base, dpi):
    plt.figure(figsize=(10,5.5)); plt.plot(tab.hour, tab.mean_bitrate_bps/1e6, marker='o', linewidth=1.2); plt.xlabel('Hora del día'); plt.ylabel('Bitrate medio (Mbit/s)'); plt.title("UGR'16 March — perfil horario — training"); plt.xticks(np.arange(24)); plt.grid(True, alpha=.25); return save_fig(base,dpi)

def plot_matrix(tab, title, base, dpi):
    m=tab.set_index('variable'); v=m.to_numpy(float); plt.figure(figsize=(7.2,6.2)); im=plt.imshow(v, aspect='auto', vmin=-1, vmax=1); plt.colorbar(im,label='Correlación de Pearson'); labels=m.columns.tolist(); plt.xticks(np.arange(len(labels)),labels,rotation=30,ha='right'); plt.yticks(np.arange(len(m.index)),m.index.tolist());
    for i in range(v.shape[0]):
        for j in range(v.shape[1]): plt.text(j,i,f'{v[i,j]:.3f}',ha='center',va='center')
    plt.title(title); return save_fig(base,dpi)

def plot_cross(tab, base, dpi):
    plt.figure(figsize=(11,5.5))
    for src,g in tab.groupby('source',sort=False): plt.plot(g.lag_minutes,g.correlation,marker='o',markersize=3,linewidth=1,label=src)
    plt.axhline(0,linewidth=.8); plt.xlabel('Adelanto source respecto a bitrate (min)'); plt.ylabel('Correlación de Pearson'); plt.title("UGR'16 March — correlación cruzada con bitrate futuro — training"); plt.legend(); plt.grid(True,alpha=.25); return save_fig(base,dpi)


def main():
    a=args_parser(); print('='*96); print('DIAGNÓSTICO ESTADÍSTICO MARCH — UGR16'); print('='*96)
    ph=require_hash(a.protocol_file, EXPECTED_PROTOCOL_SHA256, 'protocolo FROZEN')
    df, ih=load_validate(a.input); train,val=split(df)
    st=stationarity_table(train); dec=decisions(st); at=acf_table(train); pt=pacf_table(train); dp=daily_profile(train)
    vx=var_frame(train,dec['var_representation_from_protocol']); cl=corr_table(train[VAR_COLS]); cs=corr_table(vx); cc=cross_corr(vx)
    a.metrics_dir.mkdir(parents=True,exist_ok=True); a.figures_dir.mkdir(parents=True,exist_ok=True)
    paths={k:a.metrics_dir/f'{a.prefix}_{k}.csv' for k in ['stationarity','acf','pacf','daily_profile','var_correlation_levels','var_correlation_selected','var_cross_correlation']}
    report=a.metrics_dir/f'{a.prefix}_report.txt'; manifest=a.metrics_dir/f'{a.prefix}_manifest.json'
    for p in [*paths.values(),report,manifest]: reserve(p,a.overwrite)
    for name,tab in [('stationarity',st),('acf',at),('pacf',pt),('daily_profile',dp),('var_correlation_levels',cl),('var_correlation_selected',cs),('var_cross_correlation',cc)]: tab.to_csv(paths[name],index=False)
    bases=[a.figures_dir/f'{a.prefix}_{x}' for x in ['train_bitrate','first_difference','acf','pacf','daily_profile','var_correlation_levels','var_correlation_selected','var_cross_correlation']]
    for b in bases:
        for suf in ['.png','.pdf']: reserve(b.with_suffix(suf),a.overwrite)
    figs=[]; figs+=plot_series(train,bases[0],a.dpi); figs+=plot_diff(train,bases[1],a.dpi); figs+=plot_corr_lag(at,'acf',"UGR'16 March — ACF bitrate — training (1–288)",bases[2],a.dpi); figs+=plot_corr_lag(pt,'pacf',"UGR'16 March — PACF bitrate — training (1–48)",bases[3],a.dpi); figs+=plot_daily(dp,bases[4],a.dpi); figs+=plot_matrix(cl,"UGR'16 March — correlación VAR en niveles — training",bases[5],a.dpi); figs+=plot_matrix(cs,f"UGR'16 March — correlación VAR seleccionada ({dec['var_representation_from_protocol']})",bases[6],a.dpi); figs+=plot_cross(cc,bases[7],a.dpi)
    report.write_text('\n'.join(['='*96,'DIAGNÓSTICO ESTADÍSTICO MARCH — UGR16','='*96,f'Campaña: {CAMPAIGN_ID}',f'Diagnóstico: {DIAGNOSTIC_ID}',f'Protocolo SHA-256: {ph}',f'Input SHA-256: {ih}',f'Filas: total={len(df)} train={len(train)} validation={len(val)}',f'Train: {train.timestamp.min().isoformat()} -> {train.timestamp.max().isoformat()}',f'Validation: {val.timestamp.min().isoformat()} -> {val.timestamp.max().isoformat()}','Diagnósticos usan validation: NO','June usado: NO','Modelos ajustados: NO',f"Target nivel: {dec['target_level_classification']}",f"ARIMA d candidatos: {dec['arima_d_candidates_from_diagnostic']}",f"VAR representación: {dec['var_representation_from_protocol']}",'VALIDACIÓN GLOBAL: PASS','']),encoding='utf-8')
    arts=[*paths.values(),report,*figs]
    man={'campaign_id':CAMPAIGN_ID,'diagnostic_id':DIAGNOSTIC_ID,'status':'PASS','created_at':datetime.now().astimezone().isoformat(),'protocol':{'path':str(a.protocol_file),'sha256':ph,'state':'FROZEN','version':'1.0'},'input':{'path':str(a.input),'sha256':ih,'rows':len(df)},'split':{'n_train':len(train),'n_validation':len(val),'train_start':train.timestamp.min().isoformat(),'train_end':train.timestamp.max().isoformat(),'validation_start':val.timestamp.min().isoformat(),'validation_end':val.timestamp.max().isoformat(),'diagnostics_use_validation':False},'diagnostics':{'alpha':ALPHA,'acf_nlags':ACF_NLAGS,'pacf_nlags':PACF_NLAGS,'pacf_method':PACF_METHOD,'cross_corr_max_lag':CROSS_MAX_LAG},'protocol_decisions_from_diagnostics':dec,'blindness':{'june_used':False,'validation_used_for_diagnostics':False,'models_fitted':False},'software':{'python':platform.python_version(),'numpy':np.__version__,'pandas':pd.__version__,'scipy':scipy.__version__,'statsmodels':statsmodels.__version__,'scikit_learn':sklearn.__version__,'matplotlib':matplotlib.__version__},'artifacts':{str(p):sha256_file(p) for p in arts}}
    manifest.write_text(json.dumps(man,indent=2,ensure_ascii=False),encoding='utf-8')
    print(f'Total={len(df)} Train={len(train)} Validation={len(val)}'); print(f"Target={dec['target_level_classification']} ARIMA d={dec['arima_d_candidates_from_diagnostic']} VAR={dec['var_representation_from_protocol']}"); print('VALIDACIÓN GLOBAL: PASS')
    return 0

if __name__ == '__main__':
    raise SystemExit(main())
