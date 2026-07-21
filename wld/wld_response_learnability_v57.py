"""WLD v5.7 development-only perturbation-response learnability ladder."""
from __future__ import annotations
import argparse, hashlib, json, math, os
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple
import numpy as np
from wld_chromatin_modules_v55 import load_complex_module_atlas, load_v53_sparse_full_bundle, sha256_file

SCHEMA_VERSION="wld-v5.7-response-learnability-development"
REPORT_NAME="wld_v57_response_learnability_report.json"

@dataclass(frozen=True)
class ResponseLearnabilityConfig:
    reliability_replicates:int=50; null_permutations:int=200
    min_cells_per_half:int=64; max_cells_per_half:int=128
    model_seeds:Tuple[int,...]=(42,137,911)
    rank_grid:Tuple[int,...]=(1,2,4,8,16)
    alpha_grid:Tuple[float,...]=(1e-4,1e-3,1e-2,1e-1,1.,10.,100.,1000.,10000.)
    rbf_gamma_scales:Tuple[float,...]=(0.25,0.5,1.,2.,4.)
    inner_folds:int=5; route_shuffle_replicates:int=20
    response_floor:float=.002; numerical_tolerance:float=1e-6
    reliability_min_cosine:float=.20; reliability_min_positive_fraction:float=.80
    reliability_min_supported_targets:int=8; reliability_min_supported_fraction:float=.50
    ceiling_max_median_nrmse:float=.80; ceiling_min_median_cosine:float=.30
    linear_max_mean_nrmse:float=.95; linear_min_median_cosine:float=.10
    nonlinear_max_mean_nrmse:float=.90; nonlinear_max_median_nrmse:float=.95
    nonlinear_min_mean_cosine:float=.20; nonlinear_min_median_cosine:float=.10
    predictor_min_positive_target_fraction:float=.60
    def validate(self):
        if self.reliability_replicates<2 or self.null_permutations<2: raise ValueError("too few replicates")
        if self.min_cells_per_half<1 or self.max_cells_per_half<self.min_cells_per_half: raise ValueError("invalid half sizes")
        if len(set(self.model_seeds))!=len(self.model_seeds) or not self.model_seeds: raise ValueError("invalid seeds")
        if self.inner_folds<2 or min(self.rank_grid)<1 or min(self.alpha_grid)<=0: raise ValueError("invalid model grid")
LearnabilityConfig=ResponseLearnabilityConfig

def _validate_production_config(cfg):
    cfg.validate()
    if cfg.reliability_replicates<50 or cfg.null_permutations<200 or cfg.route_shuffle_replicates<20:
        raise ValueError("durable v5.7 reports require at least 50 split halves, 200 NTC nulls, and 20 topology shuffles")

def _symbol(x): return str(x or "").strip().upper()
def _hash(x): return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(",",":"),default=str).encode()).hexdigest()
def _atomic(path,value):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True); tmp=path.with_suffix(path.suffix+".tmp")
    tmp.write_text(json.dumps(value,indent=2,sort_keys=True,allow_nan=False)+"\n"); os.replace(tmp,path)
def _arr(x,name,rows=None):
    x=np.asarray(x,dtype=np.float64)
    if x.ndim!=2 or not np.isfinite(x).all() or (rows is not None and len(x)!=rows): raise ValueError(f"invalid {name}")
    return x

def deterministic_target_folds(targets,n_folds=5,seed=42,strata=None):
    targets=[_symbol(x) for x in targets]; unique=sorted(set(targets)); k=min(int(n_folds),len(unique))
    if k<2 or any(not x or x=="NTC" for x in targets): raise ValueError("at least two valid targets required")
    labels=["all"]*len(targets) if strata is None else list(map(str,strata))
    if len(labels)!=len(targets): raise ValueError("strata mismatch")
    by={}
    for t,s in zip(targets,labels):
        if t in by and by[t]!=s: raise ValueError("target crosses strata")
        by[t]=s
    assignment={}
    for s in sorted(set(labels)):
        members=sorted((t for t in unique if by[t]==s),key=lambda t:hashlib.sha256(f"{seed}|{s}|{t}".encode()).digest())
        assignment.update({t:i%k for i,t in enumerate(members)})
    return np.asarray([assignment[t] for t in targets],dtype=np.int64)

def deterministic_profile_shuffle(features,seed=42,strata=None):
    x=_arr(features,"features"); labels=["all"]*len(x) if strata is None else list(map(str,strata))
    if len(labels)!=len(x): raise ValueError("strata mismatch")
    p=np.arange(len(x))
    for s in sorted(set(labels)):
        idx=[i for i,v in enumerate(labels) if v==s]
        if len(idx)<2: continue
        idx=sorted(idx,key=lambda i:hashlib.sha256(f"{seed}|{s}|{i}".encode()).digest()); shift=1+abs(seed)%(len(idx)-1)
        while math.gcd(shift,len(idx))!=1: shift+=1
        p[np.asarray(idx)]=np.roll(np.asarray(idx),shift)
    return x[p].copy()

def _cos(p,y):
    d=np.linalg.norm(p,axis=1)*np.linalg.norm(y,axis=1)
    return np.divide(np.sum(p*y,axis=1),d,out=np.zeros(len(y)),where=d>1e-15)
def _nrmse(p,y,floor): return np.sqrt(np.mean((p-y)**2,axis=1))/np.maximum(np.sqrt(np.mean(y*y,axis=1)),floor)
def _gain(p,y,floor):
    z=np.mean(y*y,axis=1); return (z-np.mean((p-y)**2,axis=1))/np.maximum(z,floor**2)
_nrmse_rows=_nrmse
_gain_rows=_gain
def _summary(x):
    x=np.asarray(x); return {"count":int(len(x)),"mean":float(np.mean(x)) if len(x) else 0.,"median":float(np.median(x)) if len(x) else 0.,"minimum":float(np.min(x)) if len(x) else 0.,"maximum":float(np.max(x)) if len(x) else 0.}
def _metrics(p,y,targets,floor):
    n=_nrmse(p,y,floor); c=_cos(p,y); g=_gain(p,y,floor)
    return {"target_count":len(targets),"nrmse":_summary(n),"cosine":_summary(c),"relative_gain_over_zero_response":_summary(g),
            "fraction_beating_zero_response":float(np.mean(g>0)) if len(g) else 0.,"target_metrics":[{"target":t,"nrmse":float(a),"cosine":float(b),"relative_gain_over_zero_response":float(d)} for t,a,b,d in zip(targets,n,c,g)]}
def _screen_means(y,screens):
    g=np.mean(y,axis=0); return {s:np.mean(y[np.asarray(screens)==s],axis=0) for s in sorted(set(screens))},g
def _screen_matrix(screens,means,global_mean): return np.stack([means.get(s,global_mean) for s in screens])
def _basis(y,rank): return np.linalg.svd(y,full_matrices=False)[2][:min(rank,len(y),y.shape[1])]
def _standardize(x):
    m=x.mean(0); s=x.std(0); s=np.where(s>1e-12,s,1.); return (x-m)/s,m,s
def _dist(a,b): return np.maximum(np.sum(a*a,1)[:,None]+np.sum(b*b,1)[None,:]-2*a@b.T,0)

def fit_dual_ridge(x,y,alpha):
    x=_arr(x,"x"); y=_arr(y,"y",len(x)); coef=np.linalg.solve(x@x.T+float(alpha)*np.eye(len(x)),y)
    return lambda z: (_arr(z,"new_x")@x.T)@coef
def fit_rbf_kernel_ridge(x,y,alpha,gamma):
    x=_arr(x,"x"); y=_arr(y,"y",len(x)); coef=np.linalg.solve(np.exp(-gamma*_dist(x,x))+float(alpha)*np.eye(len(x)),y)
    return lambda z: np.exp(-gamma*_dist(_arr(z,"new_x"),x))@coef
def _gamma(x):
    d=_dist(x,x)[np.triu_indices(len(x),1)]; d=d[d>1e-12]
    return 1./float(np.median(d)) if len(d) else 1.

def _aggregate(y,x,targets,screens):
    order=sorted(set(targets)); yy=[]; xx=[]; ss=[]
    for target in order:
        idx=np.flatnonzero(np.asarray(targets)==target); route=x[idx]
        if np.max(np.abs(route-route[0]))>1e-10: raise ValueError(f"route changes within {target}")
        values=y[idx]
        # Exact duplicate rows are a bookkeeping duplication, not additional
        # biological evidence; retaining the first also makes equal-target
        # aggregation bitwise invariant to their multiplicity.
        yy.append(values[0].copy() if np.all(values==values[0]) else values.mean(0)); xx.append(route[0]); ss.append("|".join(sorted(set(screens[i] for i in idx))))
    return np.stack(yy),np.stack(xx),order,ss

def _rank_cv(discovery,outcomes,targets,screens,cfg,reliable=None):
    discovery=_arr(discovery,"training_discovery"); outcomes=_arr(outcomes,"training_outcomes",len(discovery))
    if discovery.shape!=outcomes.shape: raise ValueError("training split-half response shapes disagree")
    mask=np.ones(len(discovery),dtype=bool) if reliable is None else np.asarray(reliable,dtype=bool)
    if len(mask)!=len(discovery): raise ValueError("training reliability mask mismatch")
    reliable_only=bool(np.sum(mask)>=2)
    if not reliable_only: mask=np.ones(len(discovery),dtype=bool)
    discovery=discovery[mask]; outcomes=outcomes[mask]; targets=np.asarray(targets)[mask].tolist(); screens=np.asarray(screens)[mask].tolist()
    ranks=[r for r in cfg.rank_grid if r<=max(1,len(set(targets))//4)] or [1]
    folds=deterministic_target_folds(targets,cfg.inner_folds,cfg.model_seeds[0]); scores={rank:[] for rank in ranks}
    screens_array=np.asarray(screens)
    for fold in sorted(set(folds)):
        tr=folds!=fold; va=~tr; sm,g=_screen_means(discovery[tr],screens_array[tr].tolist())
        mt=_screen_matrix(screens_array[tr],sm,g); mv=_screen_matrix(screens_array[va],sm,g)
        full_basis=_basis(discovery[tr]-mt,max(ranks))
        for rank in ranks:
            b=full_basis[:rank]
            scores[rank].extend(_nrmse(mv+((discovery[va]-mv)@b.T)@b,outcomes[va],cfg.response_floor))
    records=[{"rank":rank,"training_split_a_to_b_mean_nrmse":float(np.mean(scores[rank]))} for rank in ranks]
    best=min(records,key=lambda r:(r["training_split_a_to_b_mean_nrmse"],r["rank"]))
    return int(best["rank"]),records,{"selection_targets":len(targets),"reliable_targets_only":reliable_only,"direction":"training_split_A_to_disjoint_split_B"}

def _response_preparation(y,screens,rank):
    means,g=_screen_means(y,screens); mt=_screen_matrix(screens,means,g); b=_basis(y-mt,rank)
    return means,g,b,(y-mt)@b.T

def _fit_predict(x,y,screens,new_x,new_screens,rank,kind,alpha,gamma=None,response_preparation=None):
    zx,m,s=_standardize(x); zn=(new_x-m)/s
    means,g,b,coef=response_preparation or _response_preparation(y,screens,rank)
    mn=_screen_matrix(new_screens,means,g)
    predictor=fit_dual_ridge(zx,coef,alpha) if kind=="linear" else fit_rbf_kernel_ridge(zx,coef,alpha,float(gamma))
    pred=mn+predictor(zn)@b
    pred[np.sum(np.abs(new_x),axis=1)<=1e-15]=0.
    return pred

def _response_cv_cache(y,targets,screens,rank,cfg):
    folds=deterministic_target_folds(targets,cfg.inner_folds,cfg.model_seeds[0]); cache=[]; screen_array=np.asarray(screens)
    for fold in sorted(set(folds)):
        tr=folds!=fold; va=~tr; means,g=_screen_means(y[tr],screen_array[tr].tolist())
        mt=_screen_matrix(screen_array[tr],means,g); mv=_screen_matrix(screen_array[va],means,g); b=_basis(y[tr]-mt,rank)
        cache.append({"train":tr,"validation":va,"basis":b,"coefficients":(y[tr]-mt)@b.T,
                      "validation_mean":mv,"validation_response":y[va]})
    return cache

def _model_cv(x,y,targets,screens,rank,kind,cfg,response_cv_cache=None):
    response_cache=response_cv_cache or _response_cv_cache(y,targets,screens,rank,cfg); candidates=[]; feature_cache=[]
    for fold in response_cache:
        tr=fold["train"]; va=fold["validation"]; zx,m,s=_standardize(x[tr]); zv=(x[va]-m)/s
        feature_cache.append((fold,zx,zv,_gamma(zx),np.sum(np.abs(x[va]),axis=1)<=1e-15))
    for scale in ((1.,) if kind=="linear" else cfg.rbf_gamma_scales):
        for alpha in cfg.alpha_grid:
            scores=[]
            for fold,zx,zv,base_gamma,unsupported in feature_cache:
                if kind=="linear":
                    coefficient=np.linalg.solve(zx@zx.T+float(alpha)*np.eye(len(zx)),fold["coefficients"])
                    predicted_coefficients=(zv@zx.T)@coefficient
                else:
                    gamma=base_gamma*scale; kernel=np.exp(-gamma*_dist(zx,zx))
                    coefficient=np.linalg.solve(kernel+float(alpha)*np.eye(len(zx)),fold["coefficients"])
                    predicted_coefficients=np.exp(-gamma*_dist(zv,zx))@coefficient
                p=fold["validation_mean"]+predicted_coefficients@fold["basis"]; p[unsupported]=0.
                scores.extend(_nrmse(p,fold["validation_response"],cfg.response_floor))
            candidates.append({"alpha":float(alpha),"gamma_scale":float(scale),"training_target_cv_mean_nrmse":float(np.mean(scores))})
    return min(candidates,key=lambda z:(z["training_target_cv_mean_nrmse"],z["alpha"],z["gamma_scale"])),candidates

def _route_strata(x):
    degree=np.count_nonzero(np.abs(x)>1e-15,axis=1); positive=degree[degree>0]
    cuts=np.quantile(positive,[.25,.5,.75]) if len(positive) else np.zeros(3)
    return [f"{int(d>0)}:{int(np.searchsorted(cuts,d,side='right'))}" for d in degree]

def _normalize_reliability(targets,discovery,outcomes,supported,reliability,cfg):
    rows=[]
    for i,target in enumerate(targets):
        source=dict(reliability.get(target,{})) if isinstance(reliability,Mapping) else {}
        cosine=float(source.get("median_split_half_cosine",_cos(discovery[i:i+1],outcomes[i:i+1])[0]))
        positive=float(source.get("positive_cosine_fraction",float(cosine>0)))
        rms=float(source.get("response_rms",np.sqrt(np.mean(((discovery[i]+outcomes[i])/2)**2))))
        null=float(source.get("screen_ntc_null_rms_95",0.))
        powered=bool(source.get("powered",True))
        reliable=powered and cosine>=cfg.reliability_min_cosine and positive>=cfg.reliability_min_positive_fraction and rms>null
        rows.append({"target":target,"powered":powered,"route_supported":bool(supported[i]),"median_split_half_cosine":cosine,
                     "positive_cosine_fraction":positive,"response_rms":rms,"screen_ntc_null_rms_95":null,"reliable":reliable,
                     "target_cells_per_half":source.get("target_cells_per_half"),"ntc_cells_per_half":source.get("ntc_cells_per_half")})
    supported_count=sum(r["route_supported"] for r in rows); reliable_supported=sum(r["route_supported"] and r["reliable"] for r in rows)
    fraction=reliable_supported/max(supported_count,1)
    passes=reliable_supported>=cfg.reliability_min_supported_targets and fraction>=cfg.reliability_min_supported_fraction
    return {"targets":rows,"route_supported_targets":supported_count,"reliable_route_supported_targets":reliable_supported,
            "reliable_route_supported_fraction":fraction,"passes_cohort_gate":passes,
            "gate":{"minimum_supported_targets":cfg.reliability_min_supported_targets,"minimum_supported_fraction":cfg.reliability_min_supported_fraction}}

def _subset_metrics(pred,y,targets,reliability,supported,cfg):
    masks={"all":np.ones(len(targets),dtype=bool),"reliable":np.asarray(reliability,dtype=bool),"route_supported":np.asarray(supported,dtype=bool),
           "reliable_and_route_supported":np.asarray(reliability,dtype=bool)&np.asarray(supported,dtype=bool)}
    return {name:_metrics(pred[mask],y[mask],np.asarray(targets)[mask].tolist(),cfg.response_floor) for name,mask in masks.items()}

def evaluate_learnability(train_responses,validation_discovery,validation_outcomes,train_route_features,validation_route_features,*,
                          train_targets=None,validation_targets=None,train_screens=None,validation_screens=None,reliability=None,
                          route_supported=None,train_discovery=None,train_outcomes=None,train_reliability=None,
                          historical_wld=None,config=None):
    """Evaluate the ladder on response arrays without touching any test values."""
    cfg=config or ResponseLearnabilityConfig(); cfg.validate()
    yt=_arr(train_responses,"train_responses"); train_a=_arr(yt if train_discovery is None else train_discovery,"train_discovery",len(yt))
    train_b=_arr(yt if train_outcomes is None else train_outcomes,"train_outcomes",len(yt)); ya=_arr(validation_discovery,"validation_discovery")
    yv=_arr(validation_outcomes,"validation_outcomes",len(ya)); xt=_arr(train_route_features,"train_route_features",len(yt))
    xv=_arr(validation_route_features,"validation_route_features",len(ya))
    if yt.shape[1]!=yv.shape[1] or train_a.shape!=train_b.shape or train_a.shape!=yt.shape or ya.shape!=yv.shape or xt.shape[1]!=xv.shape[1]: raise ValueError("response/route dimensions disagree")
    tt=[_symbol(x) for x in (train_targets or [f"TRAIN_{i}" for i in range(len(yt))])]
    vt=[_symbol(x) for x in (validation_targets or [f"DEV_{i}" for i in range(len(yv))])]
    ts=list(map(str,train_screens or ["screen"]*len(yt))); vs=list(map(str,validation_screens or ["screen"]*len(yv)))
    if len(tt)!=len(yt) or len(vt)!=len(yv) or len(ts)!=len(yt) or len(vs)!=len(yv): raise ValueError("metadata row mismatch")
    if set(tt)&set(vt): raise ValueError("training and development targets overlap")
    raw_tt=list(tt); raw_ts=list(ts); raw_xt=xt; raw_vt=list(vt); supplied_support=None
    if route_supported is not None:
        values=np.asarray(route_supported,dtype=bool)
        if len(values)!=len(raw_vt): raise ValueError("route support mismatch")
        supplied_support={}
        for target,value in zip(raw_vt,values):
            if target in supplied_support and supplied_support[target]!=bool(value): raise ValueError("route support changes within target")
            supplied_support[target]=bool(value)
    raw_xv=xv
    yt,xt,tt,ts=_aggregate(yt,raw_xt,raw_tt,raw_ts)
    train_a,train_ax,train_at,train_as=_aggregate(train_a,raw_xt,raw_tt,raw_ts)
    train_b,train_bx,train_bt,train_bs=_aggregate(train_b,raw_xt,raw_tt,raw_ts)
    if tt!=train_at or tt!=train_bt or ts!=train_as or ts!=train_bs or not np.allclose(xt,train_ax) or not np.allclose(xt,train_bx): raise ValueError("training split-half target order differs")
    ya,xv,vt,vs_a=_aggregate(ya,raw_xv,raw_vt,vs)
    yv,xv_b,vt_b,vs_b=_aggregate(yv,raw_xv,raw_vt,vs)
    if vt!=vt_b or vs_a!=vs_b or not np.allclose(xv,xv_b): raise ValueError("development discovery/outcome target order differs")
    vs=vs_a; supported=np.sum(np.abs(xv),axis=1)>1e-15 if supplied_support is None else np.asarray([supplied_support[t] for t in vt],dtype=bool)
    if len(supported)!=len(vt): raise ValueError("route support mismatch")
    rel=_normalize_reliability(vt,ya,yv,supported,reliability,cfg); reliable=np.asarray([r["reliable"] for r in rel["targets"]])
    train_supported=np.sum(np.abs(xt),axis=1)>1e-15
    train_rel=_normalize_reliability(tt,train_a,train_b,train_supported,train_reliability,cfg)
    train_reliable=np.asarray([r["reliable"] for r in train_rel["targets"]]); fit_mask=train_reliable&train_supported
    all_training_targets=list(tt); training_fallback=bool(np.sum(fit_mask)<2)
    if training_fallback: fit_mask=np.ones(len(tt),dtype=bool)
    rank,rank_cv,rank_audit=_rank_cv(train_a,train_b,tt,ts,cfg,train_reliable&train_supported)
    train_a_fit=train_a[fit_mask]; yt=yt[fit_mask]; xt=xt[fit_mask]; tt=np.asarray(tt)[fit_mask].tolist(); ts=np.asarray(ts)[fit_mask].tolist()
    measurement_gate=bool(train_rel["passes_cohort_gate"] and rel["passes_cohort_gate"] and not training_fallback)
    means,global_mean=_screen_means(yt,ts); generic=_screen_matrix(vs,means,global_mean)
    zero=np.zeros_like(yv); raw=ya.copy(); bin_permutation=np.random.default_rng(cfg.model_seeds[0]).permutation(yv.shape[1]); permuted=ya[:,bin_permutation]
    ceiling_means,ceiling_global=_screen_means(train_a_fit,ts); ceiling_train_mean=_screen_matrix(ts,ceiling_means,ceiling_global)
    ceiling_basis=_basis(train_a_fit-ceiling_train_mean,rank)
    lowrank_mean=_screen_matrix(vs,ceiling_means,ceiling_global); lowrank=lowrank_mean+((ya-lowrank_mean)@ceiling_basis.T)@ceiling_basis
    models={"zero_response":{"role":"persistence comparator","subsets":_subset_metrics(zero,yv,vt,reliable,supported,cfg)},
            "training_screen_mean":{"role":"training-only generic perturbation comparator","subsets":_subset_metrics(generic,yv,vt,reliable,supported,cfg)},
            "raw_split_half_oracle":{"role":"outcome-informed measurement ceiling","subsets":_subset_metrics(raw,yv,vt,reliable,supported,cfg)},
            "bin_permuted_split_half":{"role":"response-free bin-permutation negative comparator","subsets":_subset_metrics(permuted,yv,vt,reliable,supported,cfg)},
            "low_rank_split_half_oracle":{"role":"outcome-informed compressibility ceiling","selected_rank":rank,"training_only_rank_cv":rank_cv,
                                           "subsets":_subset_metrics(lowrank,yv,vt,reliable,supported,cfg)}}
    selected={}; predictions={}; response_preparation=_response_preparation(yt,ts,rank)
    response_cv_cache=_response_cv_cache(yt,tt,ts,rank,cfg)
    for kind in ("linear","rbf"):
        best,cv=_model_cv(xt,yt,tt,ts,rank,kind,cfg,response_cv_cache); zx,_,_=_standardize(xt); gamma=_gamma(zx)*best["gamma_scale"]
        pred=_fit_predict(xt,yt,ts,xv,vs,rank,kind,best["alpha"],gamma,response_preparation); predictions[kind]=pred; selected[kind]=best
        models["route_linear" if kind=="linear" else "static_nonlinear"]={"selected_training_only":best,"training_only_candidates":cv,
            "unsupported_targets_receive_zero_response":True,"subsets":_subset_metrics(pred,yv,vt,reliable,supported,cfg)}
    gate_mask=reliable&supported; shuffle_rows=[]; train_strata=_route_strata(xt); validation_strata=_route_strata(xv)
    for replicate in range(cfg.route_shuffle_replicates):
        seed=cfg.model_seeds[replicate%len(cfg.model_seeds)]+1009*replicate
        sx=deterministic_profile_shuffle(xt,seed,train_strata); sv=deterministic_profile_shuffle(xv,seed+1,validation_strata)
        for kind in ("linear","rbf"):
            # The null receives exactly the same train-only model selection as
            # the biological routes; reusing true-route choices would bias the
            # comparison in favor of the true topology.
            best,_=_model_cv(sx,yt,tt,ts,rank,kind,cfg,response_cv_cache); zx,_,_=_standardize(sx); gamma=_gamma(zx)*best["gamma_scale"]
            pred=_fit_predict(sx,yt,ts,sv,vs,rank,kind,best["alpha"],gamma,response_preparation)
            metric=_metrics(pred[gate_mask],yv[gate_mask],np.asarray(vt)[gate_mask].tolist(),cfg.response_floor)
            shuffle_rows.append({"replicate":replicate,"seed":seed,"model":kind,"selected_training_only":best,
                                 "train_profile_rows_changed":float(np.mean(np.any(np.abs(sx-xt)>1e-15,axis=1))),
                                 "development_profile_rows_changed":float(np.mean(np.any(np.abs(sv-xv)>1e-15,axis=1))),
                                 "mean_nrmse":metric["nrmse"]["mean"],"mean_cosine":metric["cosine"]["mean"]})
    all_metrics={name:value["subsets"]["all"] for name,value in models.items()}
    gate_metrics={name:value["subsets"]["reliable_and_route_supported"] for name,value in models.items()}
    # Measurement/compressibility is evaluated only where a response first
    # cleared the independently prespecified reliability gate.
    raw_gain=float(np.mean(_gain_rows(raw[reliable],yv[reliable],cfg.response_floor))) if np.any(reliable) else 0.
    low_gain=float(np.mean(_gain_rows(lowrank[reliable],yv[reliable],cfg.response_floor))) if np.any(reliable) else 0.
    reliable_low=models["low_rank_split_half_oracle"]["subsets"]["reliable"]
    ceiling_fraction=float(np.mean(_nrmse_rows(lowrank[reliable],yv[reliable],cfg.response_floor)<.90)) if np.any(reliable) else 0.
    ceiling_pass=bool(measurement_gate and np.any(reliable) and raw_gain>cfg.numerical_tolerance and reliable_low["nrmse"]["median"]<=cfg.ceiling_max_median_nrmse and reliable_low["cosine"]["median"]>=cfg.ceiling_min_median_cosine and ceiling_fraction>=.60 and low_gain>=.5*raw_gain)
    linear=gate_metrics["route_linear"]; nonlinear=gate_metrics["static_nonlinear"]
    linear_supported=models["route_linear"]["subsets"]["reliable_and_route_supported"]
    nonlinear_supported=models["static_nonlinear"]["subsets"]["reliable_and_route_supported"]
    lin_null=[r["mean_nrmse"] for r in shuffle_rows if r["model"]=="linear"]; non_null=[r["mean_nrmse"] for r in shuffle_rows if r["model"]=="rbf"]
    lin_specific=bool(linear["nrmse"]["mean"]+cfg.numerical_tolerance<float(np.quantile(lin_null,.05))); non_specific=bool(nonlinear["nrmse"]["mean"]+cfg.numerical_tolerance<float(np.quantile(non_null,.05)))
    generic=gate_metrics["training_screen_mean"]; linear_gain=linear["relative_gain_over_zero_response"]["mean"]; nonlinear_gain=nonlinear["relative_gain_over_zero_response"]["mean"]
    lin_pass=bool(linear_supported["target_count"]>0 and linear["nrmse"]["mean"]<=cfg.linear_max_mean_nrmse and linear["cosine"]["median"]>=cfg.linear_min_median_cosine and linear_supported["fraction_beating_zero_response"]>=cfg.predictor_min_positive_target_fraction and linear_gain>=.01 and linear["nrmse"]["mean"]+cfg.numerical_tolerance<generic["nrmse"]["mean"] and lin_specific)
    nonlinear_absolute=bool(nonlinear_supported["target_count"]>0 and nonlinear["nrmse"]["mean"]<=cfg.nonlinear_max_mean_nrmse and nonlinear["nrmse"]["median"]<=cfg.nonlinear_max_median_nrmse and nonlinear["cosine"]["mean"]>=cfg.nonlinear_min_mean_cosine and nonlinear["cosine"]["median"]>=cfg.nonlinear_min_median_cosine and nonlinear_supported["fraction_beating_zero_response"]>=cfg.predictor_min_positive_target_fraction and nonlinear_gain>=.01 and nonlinear["nrmse"]["mean"]+cfg.numerical_tolerance<generic["nrmse"]["mean"] and non_specific)
    nonlinear_beats_linear=nonlinear["nrmse"]["mean"]<=.98*linear["nrmse"]["mean"]
    non_pass=bool(nonlinear_absolute and (nonlinear_beats_linear or not lin_pass))
    if not measurement_gate: primary="MEASUREMENT_LIMITED"
    elif not ceiling_pass: primary="STABLE_HIGH_RANK_OR_LATENT_BOTTLENECK"
    elif lin_pass: primary="LINEAR_ROUTE_SIGNAL"
    elif non_pass: primary="STATIC_NONLINEAR_LEARNABLE"
    elif min(selected["linear"]["training_target_cv_mean_nrmse"],selected["rbf"]["training_target_cv_mean_nrmse"])<=.90: primary="WHOLE_TARGET_SHIFT_OR_OVERFIT"
    else: primary="ROUTE_PRIOR_OR_TARGET_MAPPING_INSUFFICIENT"
    flags=["TRANSIENT_RESPONSE_ONLY_NO_DYNAMICS"]
    best_route=min(linear["nrmse"]["mean"],nonlinear["nrmse"]["mean"])
    if generic["nrmse"]["mean"]<=best_route+cfg.numerical_tolerance: flags.append("TARGET_NONSPECIFIC_SCREEN_RESPONSE")
    if not (lin_specific or non_specific): flags.append("TOPOLOGY_NONSPECIFIC")
    historical_failed=bool((historical_wld or {}).get("corrected_eligibility",False) is False)
    if (lin_pass or non_pass) and historical_failed: flags.append("WLD_OPTIMIZATION_OR_PROPAGATION_FAILURE")
    next_actions={"MEASUREMENT_LIMITED":"Improve cell depth, replication, endpoint timing, or aggregation before another dynamics model.",
                  "STABLE_HIGH_RANK_OR_LATENT_BOTTLENECK":"Redesign the response representation before fitting WLD dynamics.",
                  "LINEAR_ROUTE_SIGNAL":"Use the simpler route-linear endpoint model; one endpoint does not require an ODE.",
                  "STATIC_NONLINEAR_LEARNABLE":"Use the static nonlinear route model as the next development benchmark before WLD.",
                  "WHOLE_TARGET_SHIFT_OR_OVERFIT":"Expand whole-target diversity and regularize using training targets only.",
                  "ROUTE_PRIOR_OR_TARGET_MAPPING_INSUFFICIENT":"Improve target-to-TF/complex route coverage before another WLD run."}
    sensitivity={}
    for name,value in (("route_linear",linear),("static_nonlinear",nonlinear)):
        sensitivity[name]=[{"maximum_mean_nrmse":nrmse,"minimum_median_cosine":cosine,
                            "passes":bool(value["nrmse"]["mean"]<=nrmse and value["cosine"]["median"]>=cosine)}
                           for nrmse in (.95,.90,.80) for cosine in (.10,.20,.30)]
    report={"schema_version":SCHEMA_VERSION,"created_utc":datetime.now(timezone.utc).isoformat(),"config":asdict(cfg),
            "rosters":{"training_targets":all_training_targets,"modeling_training_targets":tt,"reused_development_targets":vt,"test_targets_materialized":False},
            "reliability":{"training":train_rel,"reused_development":rel,"passes_combined_gate":measurement_gate,"training_fallback_used":training_fallback},
            "measurement_ceiling":{"selected_rank":int(rank),"training_rank_selection":rank_audit,"passes":bool(ceiling_pass),"fraction_nrmse_below_0_90":float(ceiling_fraction),"available_raw_gain":float(raw_gain),"captured_low_rank_gain":float(low_gain)},
            "models":models,"threshold_sensitivity":sensitivity,
            "matched_route_shuffles":{"replicates":cfg.route_shuffle_replicates,"evaluation_subset":"reliable_and_route_supported","rows":shuffle_rows,"linear_beats_null":bool(lin_specific),"nonlinear_beats_null":bool(non_specific)},
            "historical_wld":historical_wld or {},"decision":{"measurement_gate":measurement_gate,"low_rank_ceiling_gate":bool(ceiling_pass),"route_linear_gate":bool(lin_pass),"static_nonlinear_gate":bool(non_pass),"open_sealed_test":False},
            "diagnosis":{"primary_failure_class":primary,"flags":flags,"next_action":next_actions[primary]},
            "claims":{"development_only":True,"historical_wld_results_only":True,"fresh_wld_training":False,"test_values_materialized":False,"test_targets_evaluated":False,"confirmatory_inference":False,"digital_twin_claim":False,"attractor_claim":False}}
    validate_claims(report); return report

def validate_claims(report):
    if report.get("schema_version")!=SCHEMA_VERSION: raise ValueError("unsupported v5.7 report schema")
    claims=report.get("claims",{})
    if claims.get("development_only") is not True or claims.get("historical_wld_results_only") is not True: raise ValueError("missing development claim")
    forbidden=("fresh_wld_training","test_values_materialized","test_targets_evaluated","confirmatory_inference","digital_twin_claim","attractor_claim")
    bad=[x for x in forbidden if claims.get(x) is not False]
    if bad or report.get("decision",{}).get("open_sealed_test") is not False: raise ValueError("sealed/claim boundary crossed: "+", ".join(bad))
    provenance=report.get("provenance")
    if provenance is not None and (provenance.get("materialized_splits")!=["train","validation"] or provenance.get("test_values_materialized") is not False):
        raise ValueError("durable report provenance crossed the sealed-test boundary")
    if provenance is not None:
        boundary=provenance.get("sealed_boundary",{})
        if boundary.get("test_csr_data_or_indices_materialized") is not False or boundary.get("test_csr_row_pointer_values_materialized") is not False or boundary.get("test_metadata_fragments_field_used") is not False:
            raise ValueError("durable report lacks strict sparse-payload sealing evidence")

def _binary_mean(matrix,rows):
    rows=np.asarray(rows,dtype=np.int64)
    if not len(rows): raise ValueError("empty population")
    selected=matrix[rows].astype(np.float32,copy=True).tocsr(); selected.data.fill(1.); selected.eliminate_zeros()
    return np.asarray(selected.mean(axis=0),dtype=np.float32).ravel()
def _ordered_half(bundle,rows,seed,count):
    rows=np.asarray(rows,dtype=np.int64)
    order=sorted(rows.tolist(),key=lambda i:hashlib.sha256(f"{seed}|{int(bundle.source_rows[i])}".encode()).digest())
    return np.asarray(order[:count]),np.asarray(order[count:2*count])
def _row_normalize(x):
    x=np.asarray(x,dtype=np.float64); norm=np.linalg.norm(x,axis=1,keepdims=True)
    return np.divide(x,norm,out=np.zeros_like(x),where=norm>0)

def _load_routes(route_root,module_root):
    route_root=Path(route_root); manifest=json.loads((route_root/"route_manifest.json").read_text())
    locked=manifest.get("artifact_sha256",{}); expected=("route_vocab.json","regulator_tf_routes.npz","regulator_tf_routes.tsv.gz")
    if set(locked)!=set(expected): raise RuntimeError("TF-route manifest lacks a complete artifact lock")
    for name in expected:
        if sha256_file(route_root/name)!=locked[name]: raise RuntimeError(f"TF-route artifact hash mismatch: {name}")
    vocab=json.loads((route_root/"route_vocab.json").read_text())
    regulators=tuple(_symbol(x) for x in vocab["regulators"])
    tfs=tuple(_symbol(x) for x in vocab["tfs"])
    with np.load(route_root/"regulator_tf_routes.npz",allow_pickle=False) as z: tf=np.asarray(z["regulator_tf_support"],dtype=np.float64)
    atlas=load_complex_module_atlas(Path(module_root),verify_hashes=True)
    module_regulators=tuple(_symbol(x) for x in atlas.regulator_vocab)
    if regulators!=module_regulators or tf.shape!=(len(regulators),len(tfs)): raise RuntimeError("TF and complex route vocabularies disagree")
    complex_support=np.asarray(atlas.regulator_complex_support.toarray(),dtype=np.float64)
    dimensions={"tf_support_features":tf.shape[1],"complex_membership_features":complex_support.shape[1],
                "feature_definition":"row-normalized regulator-to-TF support plus regulator-to-complex membership",
                "downstream_response_bin_footprints_included":False,
                "route_supported_definition":"nonzero upstream TF or complex profile",
                "training_module_bundle_locks":dict(atlas.provenance.get("bundle",{}))}
    return regulators,np.concatenate([_row_normalize(tf),_row_normalize(complex_support)],axis=1),dimensions

def _target_population_response(bundle,split,target):
    responses=[]; screens=[]
    for screen in bundle.target_screens(split,target):
        tr=bundle.rows(split,screen,target); nr=bundle.rows(split,screen,"NTC")
        if len(tr) and len(nr): responses.append(_binary_mean(bundle.accessibility,tr)-_binary_mean(bundle.accessibility,nr)); screens.append(screen)
    if not responses: raise RuntimeError(f"No matched target/NTC population for {split}/{target}")
    return np.mean(np.stack(responses),axis=0).astype(np.float32),"|".join(screens)

def _split_half_data(bundle,targets,cfg,split="validation"):
    split=str(split).lower()
    if split not in {"train","validation"}: raise ValueError("split-half data may use only train or validation")
    discoveries=[]; outcomes=[]; reliability={}; null_by_screen_count={}
    def matched_ntc_null(pairs):
        key=tuple(sorted((str(screen),int(count)) for screen,count in pairs))
        if key in null_by_screen_count: return null_by_screen_count[key]
        values=[]
        for rep in range(cfg.null_permutations):
            screen_nulls=[]
            for screen,count in key:
                rows=bundle.rows(split,screen,"NTC")
                if count<1 or len(rows)<4*count: screen_nulls=[]; break
                seed=910000+rep*1009+int(_hash((split,screen))[:8],16)
                ordered=sorted(np.asarray(rows,dtype=np.int64).tolist(),key=lambda i:hashlib.sha256(f"{seed}|{int(bundle.source_rows[i])}".encode()).digest())
                groups=[np.asarray(ordered[offset*count:(offset+1)*count],dtype=np.int64) for offset in range(4)]
                first=_binary_mean(bundle.accessibility,groups[0])-_binary_mean(bundle.accessibility,groups[1])
                second=_binary_mean(bundle.accessibility,groups[2])-_binary_mean(bundle.accessibility,groups[3])
                screen_nulls.append((first+second)/2.)
            if screen_nulls: values.append(float(np.sqrt(np.mean(np.mean(np.stack(screen_nulls),axis=0)**2))))
        null_by_screen_count[key]=float(np.quantile(values,.95)) if values else float("inf")
        return null_by_screen_count[key]
    for target in targets:
        rep_a=[]; rep_b=[]; cosines=[]; rms=[]; min_target=10**9; min_ntc=10**9; used=set()
        for rep in range(cfg.reliability_replicates):
            screen_a=[]; screen_b=[]
            for screen in bundle.target_screens(split,target):
                tr=bundle.rows(split,screen,target); nr=bundle.rows(split,screen,"NTC")
                count=min(cfg.max_cells_per_half,len(tr)//2,len(nr)//4)
                if count<1: continue
                used.add((screen,count)); split_seed=int(_hash(split)[:8],16)
                ta,tb=_ordered_half(bundle,tr,1000003+split_seed+rep*7919,count); na,nb=_ordered_half(bundle,nr,2000003+split_seed+rep*7919,count)
                screen_a.append(_binary_mean(bundle.accessibility,ta)-_binary_mean(bundle.accessibility,na))
                screen_b.append(_binary_mean(bundle.accessibility,tb)-_binary_mean(bundle.accessibility,nb))
                min_target=min(min_target,count); min_ntc=min(min_ntc,count)
            if not screen_a: raise RuntimeError(f"No split-half population for {split}/{target}")
            a=np.mean(np.stack(screen_a),axis=0); b=np.mean(np.stack(screen_b),axis=0)
            rep_a.append(a); rep_b.append(b); cosines.append(float(_cos(a[None],b[None])[0])); rms.append(float(np.sqrt(np.mean(((a+b)/2)**2))))
        discoveries.append(rep_a[0]); outcomes.append(rep_b[0]); null=matched_ntc_null(used)
        reliability[target]={"powered":min_target>=cfg.min_cells_per_half and min_ntc>=cfg.min_cells_per_half,
            "target_cells_per_half":int(min_target),"ntc_cells_per_half":int(min_ntc),"median_split_half_cosine":float(np.median(cosines)),
            "positive_cosine_fraction":float(np.mean(np.asarray(cosines)>0)),"response_rms":float(np.median(rms)),"screen_ntc_null_rms_95":null,
            "replicates":cfg.reliability_replicates,"screen_count":len({screen for screen,_count in used}),
            "matched_ntc_null_keys":[f"{screen}|half={count}" for screen,count in sorted(used)]}
    reported_nulls={"+".join(f"{screen}|half={count}" for screen,count in key):float(value) for key,value in sorted(null_by_screen_count.items())}
    return np.stack(discoveries),np.stack(outcomes),reliability,reported_nulls

def _historical_summary(path):
    value=json.loads(Path(path).read_text())
    if value.get("schema_version")!="wld_v56_practical_effect_audit_v1": raise RuntimeError("unsupported v5.6 audit")
    decision=value.get("decision",{}); response=value.get("response",{}); effects=value.get("effect_summaries",{})
    source=value.get("source",{}); data=value.get("data_contract",{}); claims=value.get("claims",{})
    if source.get("source_report_immutable") is not True or data.get("development_only") is not True:
        raise RuntimeError("v5.6 audit is not immutable development-only lineage")
    forbidden_data=("test_targets_materialized","test_targets_evaluated","external_subject_study_evaluated","training_performed")
    forbidden_claims=("inference","confidence_interval_claim","p_value_claim","digital_twin_claim","attractor_claim","confirmation_claim")
    if any(data.get(name) is not False for name in forbidden_data) or any(claims.get(name) is not False for name in forbidden_claims):
        raise RuntimeError("v5.6 audit crossed a sealed or scientific-claim boundary")
    if decision.get("open_sealed_test") is not False or decision.get("eligible_to_freeze_new_confirmation_plan") is not False:
        raise RuntimeError("v5.7 requires the failed v5.6 practical gate with test closed")
    return {"source_sha256":sha256_file(Path(path)),"validation_selected_historical_comparator":True,
            "corrected_eligibility":bool(decision.get("eligible_to_freeze_new_confirmation_plan",False)),"open_sealed_test":False,
            "mean_response_nrmse":response.get("nrmse",{}).get("mean"),"mean_response_cosine":response.get("cosine",{}).get("mean"),
            "persistence_relative_gain":effects.get("persistence",{}).get("relative_gain",{}).get("mean"),"fresh_fair_wld_comparator":False}

def run_response_learnability(v53_bundle,route_root,module_root,v56_audit,output_root,config=None):
    cfg=config or ResponseLearnabilityConfig(); _validate_production_config(cfg)
    output=Path(output_root); output.mkdir(parents=True,exist_ok=True)
    inputs={"implementation_sha256":sha256_file(Path(__file__).resolve()),
            "v53_manifest_sha256":sha256_file(Path(v53_bundle)/"wld_v53_ingestion_manifest.json"),
            "split_sha256":sha256_file(Path(v53_bundle)/"whole_target_split.json"),
            "v53_matrix_sha256":sha256_file(Path(v53_bundle)/"atac_counts.GRCh38.2kb.npz"),
            "v53_cells_sha256":sha256_file(Path(v53_bundle)/"cells.tsv.gz"),
            "v53_bins_sha256":sha256_file(Path(v53_bundle)/"bins.GRCh38.2kb.tsv.gz"),
            "route_manifest_sha256":sha256_file(Path(route_root)/"route_manifest.json"),
            "route_vocab_sha256":sha256_file(Path(route_root)/"route_vocab.json"),
            "route_tensor_sha256":sha256_file(Path(route_root)/"regulator_tf_routes.npz"),
            "module_manifest_sha256":sha256_file(Path(module_root)/"complex_accessibility_module_manifest.json"),
            "module_vocab_sha256":sha256_file(Path(module_root)/"complex_accessibility_vocab.json"),
            "module_tensor_sha256":sha256_file(Path(module_root)/"complex_accessibility_modules.npz"),
            "v56_audit_sha256":sha256_file(Path(v56_audit))}
    lineage=_hash({"inputs":inputs,"config":asdict(cfg)}); final=output/REPORT_NAME
    if final.is_file():
        restored=json.loads(final.read_text())
        if restored.get("provenance",{}).get("lineage_digest")!=lineage: raise RuntimeError("existing v5.7 report has changed lineage")
        validate_claims(restored); return restored
    regulators,features,route_dims=_load_routes(route_root,module_root); index={t:i for i,t in enumerate(regulators)}
    locks=route_dims.get("training_module_bundle_locks",{})
    lock_checks={"v53_manifest_sha256":"v53_manifest_sha256","whole_target_split_sha256":"split_sha256",
                 "v53_matrix_sha256":"v53_matrix_sha256","v53_cells_sha256":"v53_cells_sha256","v53_bins_sha256":"v53_bins_sha256"}
    for locked_name,input_name in lock_checks.items():
        if locks.get(locked_name)!=inputs[input_name]: raise RuntimeError(f"v5.3 artifact no longer matches the frozen training-module lock: {locked_name}")
    v53_manifest=json.loads((Path(v53_bundle)/"wld_v53_ingestion_manifest.json").read_text())
    declared_matrix=v53_manifest.get("bundle",{}).get("matrix_sha256")
    if declared_matrix!=inputs["v53_matrix_sha256"]: raise RuntimeError("v5.3 matrix no longer matches its ingestion manifest")
    cache=output/"wld_v57_response_cache.npz"; cache_manifest=output/"wld_v57_response_cache_manifest.json"
    if cache.is_file() and cache_manifest.is_file():
        saved=json.loads(cache_manifest.read_text())
        if saved.get("lineage_digest")!=lineage or saved.get("cache_sha256")!=sha256_file(cache): raise RuntimeError("existing response cache has changed lineage")
        if saved.get("test_values_materialized") is not False: raise RuntimeError("response cache crossed the sealed-test boundary")
        with np.load(cache,allow_pickle=False) as z:
            train_y=np.asarray(z["train_responses"]); train_discovery=np.asarray(z["train_discovery"]); train_outcome=np.asarray(z["train_outcomes"])
            discovery=np.asarray(z["validation_discovery"]); outcome=np.asarray(z["validation_outcomes"])
            train_x=np.asarray(z["train_route_features"]); validation_x=np.asarray(z["validation_route_features"])
            train=tuple(map(str,z["train_targets"].tolist())); validation=tuple(map(str,z["validation_targets"].tolist()))
            train_s=list(map(str,z["train_screens"].tolist())); validation_s=list(map(str,z["validation_screens"].tolist()))
        train_reliability=dict(saved["train_reliability"]); reliability=dict(saved["validation_reliability"])
        train_nulls=dict(saved["train_ntc_null_95"]); nulls=dict(saved["validation_ntc_null_95"]); sealed_rows=int(saved["sealed_test_row_count"])
        sealed_boundary=dict(saved["sealed_boundary"])
    else:
        # A disconnect between the atomic NPZ and its manifest is recoverable:
        # neither is authoritative alone, so rebuild the derived cache from the
        # same frozen development inputs without ever loading test rows.
        bundle=load_v53_sparse_full_bundle(Path(v53_bundle),materialized_splits=("train","validation"))
        if any(str(x).lower()=="test" for x in bundle.splits) or bundle.provenance.get("test_values_materialized") is not False: raise RuntimeError("sealed test values were materialized")
        train=tuple(bundle.split_targets("train")); validation=tuple(bundle.split_targets("validation")); missing=[t for t in train+validation if t not in index]
        if missing: raise RuntimeError(f"targets missing frozen route profile: {missing}")
        responses=[]; train_s=[]
        for target in train:
            response,screen=_target_population_response(bundle,"train",target); responses.append(response); train_s.append(screen)
        train_y=np.stack(responses)
        train_discovery,train_outcome,train_reliability,train_nulls=_split_half_data(bundle,train,cfg,split="train")
        discovery,outcome,reliability,nulls=_split_half_data(bundle,validation,cfg,split="validation")
        train_x=np.stack([features[index[t]] for t in train]); validation_x=np.stack([features[index[t]] for t in validation])
        validation_s=["|".join(bundle.target_screens("validation",t)) for t in validation]; sealed_rows=int(bundle.sealed_test_row_count)
        sealed_boundary={name:bundle.provenance.get(name) for name in (
            "test_metadata_rows_read_only_for_split_integrity","test_metadata_fragments_field_used",
            "test_csr_data_or_indices_materialized","test_csr_row_pointer_values_materialized","csr_row_pointer_selection")}
        tmp=cache.with_suffix(".npz.tmp")
        with tmp.open("wb") as handle: np.savez_compressed(handle,train_responses=train_y,train_discovery=train_discovery,train_outcomes=train_outcome,
            validation_discovery=discovery,validation_outcomes=outcome,
            train_route_features=train_x,validation_route_features=validation_x,train_targets=np.asarray(train),validation_targets=np.asarray(validation),
            train_screens=np.asarray(train_s),validation_screens=np.asarray(validation_s))
        os.replace(tmp,cache)
        _atomic(cache_manifest,{"schema_version":"wld-v5.7-response-cache","lineage_digest":lineage,"cache_sha256":sha256_file(cache),
            "train_reliability":train_reliability,"validation_reliability":reliability,
            "train_ntc_null_95":{k:float(v) for k,v in train_nulls.items()},"validation_ntc_null_95":{k:float(v) for k,v in nulls.items()},"sealed_test_row_count":sealed_rows,
            "sealed_boundary":sealed_boundary,
            "materialized_splits":["train","validation"],"test_values_materialized":False})
    historical=_historical_summary(v56_audit)
    report=evaluate_learnability(train_y,discovery,outcome,train_x,validation_x,train_targets=train,validation_targets=validation,
        train_screens=train_s,validation_screens=validation_s,reliability=reliability,train_discovery=train_discovery,
        train_outcomes=train_outcome,train_reliability=train_reliability,
        route_supported=np.sum(np.abs(validation_x),axis=1)>0,historical_wld=historical,config=cfg)
    report["provenance"]={"lineage_digest":lineage,"inputs":inputs,"response_cache_sha256":sha256_file(cache),"route_dimensions":route_dims,
        "materialized_splits":["train","validation"],"sealed_test_row_count":sealed_rows,"sealed_boundary":sealed_boundary,
        "test_value_definition":"sparse accessibility data/indices payload; split-integrity metadata are audited separately",
        "test_values_materialized":False}
    report["ntc_null_95"]={"training":{k:float(v) for k,v in train_nulls.items()},"reused_development":{k:float(v) for k,v in nulls.items()}}
    validate_claims(report); _atomic(final,report); return report

def main(argv:Optional[Iterable[str]]=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v53-bundle",type=Path,required=True); parser.add_argument("--route-root",type=Path,required=True)
    parser.add_argument("--module-root",type=Path,required=True); parser.add_argument("--v56-audit",type=Path,required=True)
    parser.add_argument("--output-root",type=Path,required=True); args=parser.parse_args(list(argv) if argv is not None else None)
    report=run_response_learnability(args.v53_bundle,args.route_root,args.module_root,args.v56_audit,args.output_root)
    print("="*78); print("WLD V5.7 RESPONSE-LEARNABILITY RESULT"); print("="*78)
    print("Primary diagnosis: "+report["diagnosis"]["primary_failure_class"])
    print("Flags: "+", ".join(report["diagnosis"]["flags"])); print("Open sealed test: False")
    print("Fresh WLD training: False"); print("Attractor claim: False"); print(f"Report: {Path(args.output_root)/REPORT_NAME}")

if __name__=="__main__": main()

__all__=["ResponseLearnabilityConfig","LearnabilityConfig","deterministic_target_folds","deterministic_profile_shuffle",
         "fit_dual_ridge","fit_rbf_kernel_ridge","evaluate_learnability","validate_claims","run_response_learnability","main"]
