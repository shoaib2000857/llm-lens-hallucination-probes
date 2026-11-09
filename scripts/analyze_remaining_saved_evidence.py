#!/usr/bin/env python3
"""Trajectory-held-out drift comparison, observed label efficiency, architecture ranks."""
import csv,json,statistics
from collections import defaultdict
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'results/remaining_saved_evidence_v1'

def save(name,rows):
    if not rows:return
    with (OUT/name).open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def main():
    OUT.mkdir(exist_ok=True);summary=json.loads((ROOT/'results/trajectory_replication_v1/summary.json').read_text());drifts={(r['trajectory_seed'],r['step']):1-r['mean_linear_cka'] for r in summary['representation_drift']};grouped=defaultdict(list)
    for row in summary['rows']:
        if not row['experiment'].endswith('__frozen'):continue
        corpus=row['experiment'].split('__')[0];probe=row['probe'];seed=row['probe_seed'];reference=ROOT/'artifacts/probes'/f'{corpus}__{probe}__s{seed}__scratch_full'
        if not (reference/'evaluation.json').exists():continue
        a=json.loads((ROOT/'results/model_updates_v1'/row['experiment']/'predictions.json').read_text());b=json.loads((reference/'predictions.json').read_text())
        if {r['sample_id']:r['label'] for r in a}!={r['sample_id']:r['label'] for r in b}:raise ValueError('Recovery reference label/ID mismatch')
        auc=json.loads((reference/'evaluation.json').read_text())['metrics']['auroc'];grouped[(probe,row['trajectory_seed'],row['step'])].append(auc-row['auroc'])
    drift_rows=[];comparisons=[]
    for probe in sorted({k[0] for k in grouped}):
        points=[dict(probe=probe,trajectory_seed=t,step=s,drift=drifts[t,s],transfer_gap=statistics.mean(v),probe_seed_count=len(v)) for (p,t,s),v in grouped.items() if p==probe]
        if any(r['probe_seed_count']!=3 for r in points):raise ValueError('Incomplete probe-seed aggregation')
        drift_rows.extend(points)
        if len(points)!=9:continue
        y=np.array([r['transfer_gap'] for r in points]);traj=np.array([r['trajectory_seed'] for r in points]);d=np.array([r['drift'] for r in points]);steps=np.log1p([r['step'] for r in points])
        losses={}
        for method,x in [('drift',d),('log_training_steps',steps)]:
            predicted=np.empty(len(y))
            for seed in [42,43,44]:
                train=traj!=seed;test=~train;fit=LinearRegression().fit(x[train,None],y[train]);predicted[test]=fit.predict(x[test,None])
            losses[method]=mean_squared_error(y,predicted)
        comparisons.append(dict(probe=probe,n_checkpoints=len(points),trajectory_groups=3,spearman_drift_gap=float(spearmanr(d,y).statistic),spearman_steps_gap=float(spearmanr(steps,y).statistic),heldout_trajectory_mse_drift=losses['drift'],heldout_trajectory_mse_steps=losses['log_training_steps'],drift_mse_improvement=losses['log_training_steps']-losses['drift']))
    save('drift_gap_points.csv',drift_rows);save('drift_vs_steps.csv',comparisons)
    curves=[];attainment=[]
    for (probe,t,step),_ in grouped.items():
        prefix='updates_v1' if t==42 else f'updates_v1r{t}';corpus=f'{prefix}_qwen3_8b_step{step:06d}'
        refs=[ROOT/'artifacts/probes'/f'{corpus}__{probe}__s{s}__scratch_full/evaluation.json' for s in [42,43,44]]
        if not all(p.exists() for p in refs):continue
        baseline=statistics.mean(json.loads(p.read_text())['metrics']['auroc'] for p in refs)
        for method in ['scratch','warm']:
            observed=[]
            for budget in ['25','50','100','250','500','1000','full']:
                folders=[ROOT/'artifacts/probes'/f'{corpus}__{probe}__s{s}__{method}_{budget}' for s in [42,43,44]]
                if not all((p/'evaluation.json').exists() and (p/'complete.json').exists() for p in folders):continue
                scores=[json.loads((p/'evaluation.json').read_text())['metrics']['auroc'] for p in folders];labels=[json.loads((p/'config.json').read_text())['total_target_labels'] for p in folders];times=[json.loads((p/'complete.json').read_text())['seconds_this_invocation'] for p in folders]
                row=dict(corpus=corpus,probe=probe,method=method,budget=budget,label_count_mean=statistics.mean(labels),auroc_mean=statistics.mean(scores),auroc_seed_std=statistics.stdev(scores),recorded_fit_seconds_mean=statistics.mean(times),reference_auroc=baseline);curves.append(row);observed.append(row)
            for fraction in [.9,.95]:
                threshold=.5+fraction*(baseline-.5) if baseline>.52 else None;met=[r for r in observed if threshold is not None and r['auroc_mean']>=threshold]
                attainment.append(dict(corpus=corpus,probe=probe,method=method,fraction_of_above_chance_signal=fraction,reference_auroc=baseline,target_auroc=threshold,minimum_observed_labels=min((r['label_count_mean'] for r in met),default=None),budgets_observed=';'.join(r['budget'] for r in observed)))
    save('label_efficiency_curves.csv',curves);save('label_efficiency_attainment.csv',attainment)
    rank=[]
    for probe in ['mean_linear','last_linear','last_mlp','M0','M1','M2','M3','M4']:
        row={'probe':probe}
        for source in ['ministral3_8b','qwen3_8b']:
            paths=[ROOT/'artifacts/probes'/f'{source}__{probe}__s{s}__scratch_full/evaluation.json' for s in [42,43,44]]
            row[source+'_auroc']=statistics.mean(json.loads(p.read_text())['metrics']['auroc'] for p in paths)
        rank.append(row)
    save('architecture_rank_comparison.csv',rank)
    rho=float(spearmanr([r['ministral3_8b_auroc'] for r in rank],[r['qwen3_8b_auroc'] for r in rank]).statistic)
    (OUT/'summary.json').write_text(json.dumps(dict(drift_comparisons=comparisons,architecture_auroc_rank_spearman=rho,label_curve_rows=len(curves),attainment_rows=len(attainment)),indent=2))
    (OUT/'README.md').write_text('Exploratory nine-checkpoint comparison with three optimizer trajectories sharing one parent, domain and adaptation data. Average probe seeds before trajectory-held-out regression; not nine independent domains. Transfer gap uses matched target-trained reference labels/IDs. No causal or broadly validated drift predictor claim. Label efficiency is observed budgets only, with90/95percent of above-chance reference AUROC, no interpolation; references near chance are undefined. Fit times are recorded invocation durations, potentially lower bounds after resume. Seed mean attainment is descriptive without uncertainty on crossing. Architecture rank is two families on source SQuAD, not universal ranking. All labels are judge consensus.\n')
    print(json.dumps(dict(drift_probes=len(comparisons),label_curve_rows=len(curves),architecture_rank_spearman=rho)),flush=True)
if __name__=='__main__':main()
