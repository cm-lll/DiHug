"""Analytic limit: ranking is not a universally proper probability score."""
import math,json
from pathlib import Path
out=Path(__file__).resolve().parents[2]/'results/local_allocation_design_20261108'
# At a fixed observable Xt, exactly one of two legal edges is present in X0.
# P(Y=(1,0)|Xt)=q. CE optimum marginals=(q,1-q).
q=.8;d_ce=2*math.log(q/(1-q));d_rank=math.log(q/(1-q))
def softplus(x):return max(x,0)+math.log1p(math.exp(-abs(x)))
def risk(d):return q*softplus(-d)+(1-q)*softplus(d)
derivative=1/(1+math.exp(-d_ce))-q
assert derivative>0 and risk(d_rank)<risk(d_ce)
(out/'calibration_limit.json').write_text(json.dumps({'q':q,'CE_optimal_marginals':[q,1-q],'CE_optimal_score_difference':d_ce,'ranking_optimal_score_difference':d_rank,'ranking_risk_at_CE_optimum':risk(d_ce),'ranking_risk_at_rank_optimum':risk(d_rank),'ranking_derivative_at_CE_optimum':derivative,'meaning':'dependent labels can give different population optima; this ranking term does not preserve calibrated posterior probabilities in general','scope':'mathematical construction, not observed PubMed root cause','model_calls':0},indent=2))
