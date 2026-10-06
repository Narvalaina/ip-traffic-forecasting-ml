#!/usr/bin/env python3
"""
TREE-STABILITY-001

Análisis de estabilidad entre semillas de los representantes congelados de los
métodos de conjunto basados en árboles sobre UGR'16 April Week #3.

No accede a June. La selección RF*/XGB*/LGBM*/TREE* no puede cambiar.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import multiprocessing as mp
import os
import pickle
import platform
import sys
import time
import traceback
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

CAMPAIGN_ID = "TREE-STABILITY-001"
PARENT_CAMPAIGN_ID = "UGR16-TREE-ENSEMBLES-001"

AP = Path("data/processed/ugr16/april_week3_prepared_5min.parquet")
P017 = Path("docs/project_governance/017_tree_ensembles_protocol_2026-08-18.md")
P018 = Path("docs/project_governance/018_tree_march_screen_closure_2026-08-19.md")
P019 = Path("docs/project_governance/019_tree_freeze_2026-08-19.md")
ENV = Path("results/metrics/tree_ensembles/tree_environment_preflight_20260819.txt")
APR_MET = Path("results/metrics/tree_ensembles/ugr16_tree_april_selection_candidate_horizon_metrics.csv")
APR_SCO = Path("results/metrics/tree_ensembles/ugr16_tree_april_selection_candidate_scores.csv")
APR_WIN = Path("results/metrics/tree_ensembles/ugr16_tree_april_selection_family_winners.csv")
APR_TREE = Path("results/metrics/tree_ensembles/ugr16_tree_april_selection_tree_winner.csv")
APR_PRED = Path("results/predictions/tree_ensembles/ugr16_tree_april_selection_predictions.parquet")

SHA = {
    "017": "8c0a94e4ee80a84b78bf077d6abc3c18617ad1736b6d978047d0db1d52055003",
    "018": "53445e2f95fcdc74a66240c5368cc5ad64d826e0a3300d95a61da0c91ca94e66",
    "019": "cdf5f1b46f183b19835d7d0159292a129c3c1f621894625627ea2fda0cd24ae3",
    "env": "82ab24602f66975929904bab802da8893bd059e51c7214bccf412a95275216b2",
    "april": "4321f783724b64cbd6ade37e9b6762e540d9df35df9ffd3bf2d020c8156034b0",
    "metrics": "a86538877b12fc4f84502e1a35c075c54ac33d2554fccf2bc8bfbdba5de87466",
    "scores": "5cc5808e59480446b20167e376e38338d732106d3e573a2e0e852ea502208e79",
    "winners": "4a11350a291e65af561a858da529549c9ee3032ececef311640b110a42e8f3be",
    "tree": "25ccd64176f5acd9c134ae0b949d2aa7d835766ba89aabc274be5a60f18eff7a",
    "predictions": "25e205431d7eee5e0e5dbefb9dc5f8ab7bb0d384795042c990efa2e67060c947",
}

TARGET = "bitrate_bps"
HORIZONS = [1, 3, 6, 12]
HMIN = {1: 5, 3: 15, 6: 30, 12: 60}
LAGS = [1,2,3,4,5,6,7,8,9,10,11,12,24,72,144,288]
FEATURES = ["current_value", *[f"lag_{x}" for x in LAGS],
            "hour_sin","hour_cos","dow_sin","dow_cos","is_weekend","trend"]
TRAIN_END = 1401
CANONICAL = 20260818
SEEDS = [20260818, 20260819, 20260820]
NEW_SEEDS = [20260819, 20260820]
N_JOBS = 2
TIMEOUT = 600
COUNTS = {1: 602, 3: 600, 6: 597, 12: 591}
REPS = {"random_forest":"RF01","xgboost":"XGB06","lightgbm":"LGB01"}
TREE_STAR = ("lightgbm","LGB01")

CFG = {
"RF01": dict(n_estimators=200,max_depth=6,min_samples_leaf=10,max_features=0.7),
"XGB06": dict(n_estimators=300,learning_rate=0.10,max_depth=3,min_child_weight=5,
              subsample=0.8,colsample_bytree=0.8,reg_lambda=5),
"LGB01": dict(n_estimators=200,learning_rate=0.03,num_leaves=7,max_depth=3,
              min_child_samples=30,subsample=1.0,colsample_bytree=1.0,reg_lambda=1),
}

def parse_args():
    p=argparse.ArgumentParser()
    p.add_argument("--preflight-only",action="store_true")
    p.add_argument("--overwrite",action="store_true")
    return p.parse_args()

def digest(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda:f.read(1<<20),b""): h.update(b)
    return h.hexdigest()

def check(path, expected, label):
    path=Path(path)
    if not path.is_file(): raise FileNotFoundError(f"{label}: {path}")
    got=digest(path)
    if got!=expected: raise RuntimeError(f"{label} SHA mismatch: {got}")
    return got

def check_sidecar(path,label):
    side=Path(str(path)+".sha256")
    if not side.is_file(): raise RuntimeError(f"{label}: falta sidecar")
    if side.read_text(encoding="utf-8").split()[0] != digest(path):
        raise RuntimeError(f"{label}: sidecar inválido")

def build_features(df):
    ts=df["timestamp"]; y=df[TARGET].astype(float)
    X=pd.DataFrame(index=df.index)
    X["current_value"]=y
    for lag in LAGS: X[f"lag_{lag}"]=y.shift(lag)
    md=(ts.dt.hour*60+ts.dt.minute).astype(float)
    dow=ts.dt.dayofweek.astype(float)
    X["hour_sin"]=np.sin(2*np.pi*md/(24*60))
    X["hour_cos"]=np.cos(2*np.pi*md/(24*60))
    X["dow_sin"]=np.sin(2*np.pi*dow/7)
    X["dow_cos"]=np.cos(2*np.pi*dow/7)
    X["is_weekend"]=(ts.dt.dayofweek>=5).astype(float)
    X["trend"]=np.arange(len(df),dtype=float)
    if list(X.columns)!=FEATURES or len(X.columns)!=23:
        raise RuntimeError("Features no coinciden con el freeze")
    return X

def supervised(X,df,h):
    z=X.copy()
    z["origin_index"]=np.arange(len(df))
    z["target_index"]=z["origin_index"]+h
    z["origin_timestamp"]=df["timestamp"]
    z["target_timestamp"]=df["timestamp"].shift(-h)
    z["target_value"]=df[TARGET].shift(-h)
    z["persistence_value"]=df[TARGET]
    z=z.dropna().reset_index(drop=True)
    tr=z[z["target_index"]<=TRAIN_END].copy()
    va=z[z["origin_index"]>=TRAIN_END].copy()
    if len(va)!=COUNTS[h]: raise RuntimeError(f"H{h}: coverage")
    if int(tr["target_index"].max())>int(va["origin_index"].min()):
        raise RuntimeError(f"H{h}: leakage")
    return tr,va

def model_for(family,cid,seed):
    p=CFG[cid]
    if family=="random_forest":
        return RandomForestRegressor(criterion="absolute_error",bootstrap=True,
            random_state=seed,n_jobs=N_JOBS,**p)
    if family=="xgboost":
        return XGBRegressor(booster="gbtree",tree_method="hist",
            objective="reg:absoluteerror",gamma=0,reg_alpha=0,
            random_state=seed,n_jobs=N_JOBS,verbosity=0,**p)
    if family=="lightgbm":
        return LGBMRegressor(boosting_type="gbdt",objective="regression_l1",
            reg_alpha=0,random_state=seed,n_jobs=N_JOBS,verbosity=-1,
            subsample_freq=0,**p)
    raise RuntimeError(family)

def worker(conn,family,cid,seed,Xt,yt,Xv):
    try:
        m=model_for(family,cid,seed)
        t=time.perf_counter(); m.fit(Xt,yt); fit=time.perf_counter()-t
        t=time.perf_counter(); pred=np.asarray(m.predict(Xv),float); pr=time.perf_counter()-t
        size=len(pickle.dumps(m,protocol=pickle.HIGHEST_PROTOCOL))
        conn.send(dict(status="PASS",pred=pred,fit_seconds=fit,predict_seconds=pr,
                       serialized_model_bytes=size,negative_predictions=int((pred<0).sum())))
    except BaseException as e:
        conn.send(dict(status="ERROR",error_type=type(e).__name__,
                       error_message=str(e),traceback=traceback.format_exc()))
    finally: conn.close()

def fit_timeout(family,cid,seed,Xt,yt,Xv):
    ctx=mp.get_context("spawn"); parent,child=ctx.Pipe(duplex=False)
    proc=ctx.Process(target=worker,args=(child,family,cid,seed,Xt,yt,Xv))
    proc.start(); child.close(); proc.join(TIMEOUT)
    if proc.is_alive():
        proc.terminate(); proc.join(10)
        if proc.is_alive(): proc.kill(); proc.join()
        parent.close(); return {"status":"RESOURCE_LIMIT"}
    if not parent.poll(2):
        code=proc.exitcode; parent.close()
        return {"status":"ERROR","error_type":"WorkerNoResult","error_message":f"exitcode={code}"}
    out=parent.recv(); parent.close(); return out

def smape(y,p):
    d=np.abs(y)+np.abs(p)
    r=np.divide(2*np.abs(y-p),d,out=np.zeros_like(y,float),where=d!=0)
    return float(100*np.mean(r))

def atomic_csv(df,path):
    tmp=path.with_name("."+path.name+".tmp"); df.to_csv(tmp,index=False); os.replace(tmp,path)
def atomic_parquet(df,path):
    tmp=path.with_name("."+path.name+".tmp"); df.to_parquet(tmp,index=False); os.replace(tmp,path)
def atomic_text(s,path):
    tmp=path.with_name("."+path.name+".tmp"); tmp.write_text(s,encoding="utf-8"); os.replace(tmp,path)
def atomic_json(obj,path):
    tmp=path.with_name("."+path.name+".tmp")
    tmp.write_text(json.dumps(obj,ensure_ascii=False,indent=2)+"\n",encoding="utf-8"); os.replace(tmp,path)

def pkg(name):
    try:return version(name)
    except PackageNotFoundError:return "NOT_INSTALLED"

def main():
    args=parse_args(); start=time.perf_counter()

    check(P017,SHA["017"],"017"); check_sidecar(P017,"017")
    check(P018,SHA["018"],"018"); check_sidecar(P018,"018")
    check(P019,SHA["019"],"019"); check_sidecar(P019,"019")
    check(ENV,SHA["env"],"environment"); check_sidecar(ENV,"environment")
    check(AP,SHA["april"],"April")
    check(APR_MET,SHA["metrics"],"April metrics")
    check(APR_SCO,SHA["scores"],"April scores")
    check(APR_WIN,SHA["winners"],"April winners")
    check(APR_TREE,SHA["tree"],"April TREE*")
    check(APR_PRED,SHA["predictions"],"April predictions")

    freeze=P019.read_text(encoding="utf-8")
    for token in ["Frozen = TRUE","RF*   = RF01","XGB*  = XGB06","LGBM* = LGB01",
                  "TREE* = LGB01 / LightGBM","20260818","20260819","20260820"]:
        if token not in freeze: raise RuntimeError(f"019 sin token: {token}")

    df=pd.read_parquet(AP)[["timestamp",TARGET]].copy()
    df["timestamp"]=pd.to_datetime(df["timestamp"],errors="raise")
    df=df.reset_index(drop=True)
    if len(df)!=2004 or df["timestamp"].iloc[0]!=pd.Timestamp("2016-04-11 01:00:00")        or df["timestamp"].iloc[-1]!=pd.Timestamp("2016-04-17 23:55:00"):
        raise RuntimeError("April no coincide con la captura congelada")

    winners=pd.read_csv(APR_WIN)
    if dict(zip(winners["family"],winners["candidate_id"]))!=REPS:
        raise RuntimeError("Representantes no coinciden con 019")
    tw=pd.read_csv(APR_TREE)
    if len(tw)!=1 or (str(tw.iloc[0]["family"]),str(tw.iloc[0]["candidate_id"]))!=TREE_STAR:
        raise RuntimeError("TREE* no coincide con 019")

    X=build_features(df)
    train={}; valid={}
    for h in HORIZONS: train[h],valid[h]=supervised(X,df,h)

    canonical_pred=pd.read_parquet(APR_PRED)
    canonical_pred=canonical_pred[canonical_pred["candidate_id"].isin(REPS.values())].copy()
    if set(canonical_pred["seed"])!={CANONICAL}: raise RuntimeError("Seed canónica inesperada")
    if len(canonical_pred)!=3*sum(COUNTS.values()): raise RuntimeError("Canonical coverage")

    print("="*100)
    print("TREE-STABILITY-001 — PREFLIGHT")
    print("="*100)
    print(f"017:                         PASS | {SHA['017']}")
    print(f"018:                         PASS | {SHA['018']}")
    print(f"019:                         PASS | {SHA['019']}")
    print(f"Environment:                 PASS | {SHA['env']}")
    print(f"April:                       PASS | 2004 filas | SHA={SHA['april']}")
    print("Representantes:              PASS | RF01 / XGB06 / LGB01")
    print("TREE*:                       PASS | LGB01 / LightGBM")
    print("Seeds:                       20260818 / 20260819 / 20260820")
    print("Seed canónica reutilizada:   20260818")
    print("Nuevos fits:                 24")
    print("Selección modificable:       NO")
    print("Umbral post hoc:             NO")
    print("Refit validation:            NO")
    print("June accessible by runner:   NO")
    for h in HORIZONS:
        print(f"H{h:<2}: train={len(train[h]):,} | validation={len(valid[h]):,} | leakage=PASS")
    print("TREE-STABILITY PREFLIGHT: PASS")
    print("TREE-JUNE-BLIND-001: NOT RUN")
    if args.preflight_only:return 0

    mdir=Path("results/metrics/tree_ensembles"); pdir=Path("results/predictions/tree_ensembles")
    prefix="ugr16_tree_stability"
    paths={
      "horizon_metrics":mdir/f"{prefix}_horizon_metrics.csv",
      "seed_scores":mdir/f"{prefix}_seed_scores.csv",
      "summary":mdir/f"{prefix}_summary.csv",
      "runtime":mdir/f"{prefix}_runtime.csv",
      "predictions":pdir/f"{prefix}_predictions.parquet",
      "report":mdir/f"{prefix}_report.txt",
      "manifest":mdir/f"{prefix}_manifest.json",
    }
    existing=[str(p) for p in paths.values() if p.exists()]
    if existing and not args.overwrite: raise FileExistsError("\n".join(existing))
    mdir.mkdir(parents=True,exist_ok=True); pdir.mkdir(parents=True,exist_ok=True)

    apr_metrics=pd.read_csv(APR_MET)
    canon_met=apr_metrics[apr_metrics["candidate_id"].isin(REPS.values())].copy()
    if len(canon_met)!=12: raise RuntimeError("12 métricas canónicas esperadas")
    canon_met["source"]="reused_april_selection"

    ccols=["capture_key","capture_name","evaluation_role","family","candidate_id","seed",
           "horizon_steps","horizon_minutes","origin_index","target_index","origin_timestamp",
           "target_timestamp","y_true_bps","y_pred_bps","persistence_pred_bps","error_bps",
           "absolute_error_bps"]
    canonical_pred["evaluation_role"]="stability"
    canonical_pred["source"]="reused_april_selection"
    canonical_pred=canonical_pred[ccols+["source"]]

    mase=float(np.mean(np.abs(np.diff(df.loc[:TRAIN_END,TARGET].to_numpy(float)))))
    metric_rows=[]; runtime_rows=[]; pred_rows=[]; total=24; done=0

    for seed in NEW_SEEDS:
        print("\n"+"="*100); print(f"SEED {seed}"); print("="*100)
        for family,cid in REPS.items():
            print(f"\n{family} / {cid}")
            for h in HORIZONS:
                done+=1; tr=train[h]; va=valid[h]
                Xt=tr[FEATURES].to_numpy(float); yt=tr["target_value"].to_numpy(float)
                Xv=va[FEATURES].to_numpy(float); y=va["target_value"].to_numpy(float)
                pers=va["persistence_value"].to_numpy(float)
                print(f"  H{h:<2} [{done}/{total}] ... ",end="",flush=True)
                r=fit_timeout(family,cid,seed,Xt,yt,Xv)
                rr=dict(family=family,candidate_id=cid,seed=seed,horizon_steps=h,
                        horizon_minutes=HMIN[h],status=r["status"],fit_seconds=np.nan,
                        predict_seconds=np.nan,serialized_model_bytes=np.nan,negative_predictions=np.nan)
                if r["status"]=="RESOURCE_LIMIT":
                    runtime_rows.append(rr); print("RESOURCE_LIMIT"); continue
                if r["status"]!="PASS":
                    raise RuntimeError(f"{cid} seed={seed} H{h}: {r.get('error_type')} {r.get('error_message')}")
                yp=np.asarray(r["pred"],float)
                rr.update(fit_seconds=r["fit_seconds"],predict_seconds=r["predict_seconds"],
                          serialized_model_bytes=r["serialized_model_bytes"],
                          negative_predictions=r["negative_predictions"])
                runtime_rows.append(rr)
                e=y-yp; pe=y-pers; mae=float(np.mean(np.abs(e))); pmae=float(np.mean(np.abs(pe)))
                metric_rows.append(dict(
                    family=family,candidate_id=cid,seed=seed,horizon_steps=h,horizon_minutes=HMIN[h],
                    n_predictions=len(va),model_mae_bps=mae,model_mae_mbps=mae/1e6,
                    persistence_mae_bps=pmae,persistence_mae_mbps=pmae/1e6,
                    mae_ratio_vs_persistence=mae/pmae,skill_vs_persistence_pct=100*(pmae-mae)/pmae,
                    model_rmse_bps=float(np.sqrt(np.mean(e**2))),model_smape_pct=smape(y,yp),
                    model_mase=mae/mase,model_bias_bps=float(np.mean(e)),
                    model_underprediction_pct=float(100*np.mean(e>0)),
                    model_p95_absolute_error_bps=float(np.quantile(np.abs(e),0.95)),
                    negative_predictions=int((yp<0).sum()),coverage=1.0,source="stability_refit"))
                vr=va.reset_index(drop=True)
                for i in range(len(vr)):
                    pred_rows.append(dict(
                        capture_key="april",capture_name="UGR'16 April Week #3",evaluation_role="stability",
                        family=family,candidate_id=cid,seed=seed,horizon_steps=h,horizon_minutes=HMIN[h],
                        origin_index=int(vr.at[i,"origin_index"]),target_index=int(vr.at[i,"target_index"]),
                        origin_timestamp=vr.at[i,"origin_timestamp"],target_timestamp=vr.at[i,"target_timestamp"],
                        y_true_bps=float(y[i]),y_pred_bps=float(yp[i]),persistence_pred_bps=float(pers[i]),
                        error_bps=float(e[i]),absolute_error_bps=float(abs(e[i])),source="stability_refit"))
                print(f"PASS | ratio={mae/pmae:.6f}")

    extra_met=pd.DataFrame(metric_rows); runtime=pd.DataFrame(runtime_rows); extra_pred=pd.DataFrame(pred_rows)

    cols=list(extra_met.columns)
    missing=[c for c in cols if c not in canon_met.columns]
    if missing: raise RuntimeError("Métricas canónicas sin columnas: "+",".join(missing))
    canon_met=canon_met[cols]
    hmetrics=pd.concat([canon_met,extra_met],ignore_index=True).sort_values(
        ["family","candidate_id","seed","horizon_steps"]).reset_index(drop=True)
    predictions=pd.concat([canonical_pred,extra_pred],ignore_index=True).sort_values(
        ["family","candidate_id","seed","horizon_steps","target_timestamp"]).reset_index(drop=True)

    scores=(hmetrics.groupby(["family","candidate_id","seed"],as_index=False)
            ["mae_ratio_vs_persistence"].mean().rename(columns={"mae_ratio_vs_persistence":"score"}))
    scores["selected_before_stability"]=True; scores["selection_changed"]=False

    srows=[]
    for (family,cid),g in scores.groupby(["family","candidate_id"],sort=True):
        vals=g["score"].to_numpy(float)
        canonical=float(g.loc[g["seed"].eq(CANONICAL),"score"].iloc[0])
        srows.append(dict(family=family,candidate_id=cid,tree_star=(family,cid)==TREE_STAR,
                          n_seeds=len(vals),canonical_score=canonical,mean_score=float(vals.mean()),
                          std_score_population=float(vals.std(ddof=0)),min_score=float(vals.min()),
                          max_score=float(vals.max()),score_range=float(vals.max()-vals.min()),
                          max_abs_delta_vs_canonical=float(np.max(np.abs(vals-canonical))),
                          max_relative_delta_vs_canonical_pct=float(100*np.max(np.abs(vals-canonical))/abs(canonical)),
                          selection_changed=False))
    summary=pd.DataFrame(srows)

    checks={
      "horizon_metrics_36":len(hmetrics)==36,
      "seed_scores_9":len(scores)==9,
      "summary_3":len(summary)==3,
      "runtime_24":len(runtime)==24,
      "predictions_21510":len(predictions)==21510,
      "only_april":set(predictions["capture_key"])=={"april"},
      "three_representatives":set(predictions["candidate_id"])==set(REPS.values()),
      "three_seeds":set(predictions["seed"])==set(SEEDS),
      "all_horizons":set(predictions["horizon_steps"])==set(HORIZONS),
      "finite_predictions":np.isfinite(predictions["y_pred_bps"].to_numpy(float)).all(),
      "coverage_one":hmetrics["coverage"].eq(1.0).all(),
      "no_resource_limit":not runtime["status"].eq("RESOURCE_LIMIT").any(),
      "no_error":not runtime["status"].eq("ERROR").any(),
      "selection_changed_false":not scores["selection_changed"].any(),
      "tree_star_lgb01":summary.loc[summary["tree_star"],"candidate_id"].tolist()==["LGB01"],
    }
    if not all(checks.values()):
        raise RuntimeError("Validaciones fallidas: "+",".join(k for k,v in checks.items() if not v))

    report=[
      "="*108,"TREE-STABILITY-001 — ESTABILIDAD ENTRE SEMILLAS","="*108,
      "RF* = RF01 | XGB* = XGB06 | LGBM* = LGB01 | TREE* = LGB01 / LightGBM",
      "Seeds = 20260818 / 20260819 / 20260820",
      "Selección modificable = NO","Umbral post hoc de estabilidad = NO","",
      "SCORES POR SEMILLA","-"*108,
      scores.to_string(index=False,float_format=lambda x:f"{x:.9f}"),"",
      "RESUMEN DESCRIPTIVO","-"*108,
      summary.to_string(index=False,float_format=lambda x:f"{x:.9f}"),"",
      "VALIDACIONES","-"*108,
    ]
    report += [f"{k:42s}: {'PASS' if v else 'FAIL'}" for k,v in checks.items()]
    report += ["","TREE-JUNE-BLIND-001 = NOT RUN",""]
    report="\n".join(report)

    atomic_csv(hmetrics,paths["horizon_metrics"])
    atomic_csv(scores,paths["seed_scores"])
    atomic_csv(summary,paths["summary"])
    atomic_csv(runtime,paths["runtime"])
    atomic_parquet(predictions,paths["predictions"])
    atomic_text(report,paths["report"])

    outputs={}
    for k,p in paths.items():
        if k=="manifest": continue
        outputs[k]={"path":str(p.resolve()),"sha256":digest(p),"bytes":p.stat().st_size}
    manifest={
      "campaign_id":CAMPAIGN_ID,"parent_campaign_id":PARENT_CAMPAIGN_ID,"status":"PASS",
      "capture":"april","june_accessed":False,"representatives":REPS,
      "tree_star":{"family":"lightgbm","candidate_id":"LGB01"},
      "canonical_seed":CANONICAL,"stability_seeds":SEEDS,"additional_seeds_executed":NEW_SEEDS,
      "canonical_seed_reused":True,"selection_can_change":False,
      "posthoc_stability_threshold":None,"validation":{k:bool(v) for k,v in checks.items()},
      "software":{"python":platform.python_version(),"numpy":np.__version__,"pandas":pd.__version__,
                  "scikit_learn":pkg("scikit-learn"),"xgboost":pkg("xgboost-cpu"),
                  "lightgbm":pkg("lightgbm"),"pyarrow":pkg("pyarrow")},
      "processing_seconds":time.perf_counter()-start,"representatives_changed":False,
      "june_blind_run":False,"outputs":outputs}
    atomic_json(manifest,paths["manifest"])

    print("\n"+report)
    print(f"Manifiesto: {paths['manifest'].resolve()}")
    print("TREE-STABILITY-001: PASS")
    print("REPRESENTATIVES CHANGED: NO")
    print("TREE-JUNE-BLIND-001: NOT RUN")
    return 0

if __name__=="__main__":
    mp.freeze_support()
    raise SystemExit(main())
