#!/usr/bin/env python3
"""Code-derived parameter counts, layer-order diagnostic and unified vector figures."""
import csv,html,json,sys
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from portability.probe_training import make_probe,POOLED,RAW
from portability.storage import atomic_json

def diagram(path,title,rows):
    width=1200;height=90+len(rows)*112
    parts=[f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">','<defs><marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto"><path d="M0 0 L10 5 L0 10Z" fill="#596579"/></marker></defs>',f'<rect width="{width}" height="{height}" fill="white"/>',f'<text x="24" y="36" font-family="sans-serif" font-size="23" fill="#243047">{html.escape(title)}</text>']
    for r,row in enumerate(rows):
        y=70+r*112
        for c,lines in enumerate(row):
            x=24+c*236
            parts.append(f'<rect x="{x}" y="{y}" width="214" height="84" rx="10" fill="{["#e9eff9","#e7f4ef","#f2ecfa","#fff0e6","#e9eff9"][c]}" stroke="#8694aa"/>')
            for j,line in enumerate(lines.split('\n')):
                parts.append(f'<text x="{x+107}" y="{y+27+j*21}" text-anchor="middle" font-family="sans-serif" font-size="15" fill="#243047">{html.escape(line)}</text>')
            if c<4:parts.append(f'<path d="M{x+214} {y+42} H{x+235}" stroke="#596579" stroke-width="2" marker-end="url(#arrow)"/>')
    parts.append('</svg>');path.write_text('\n'.join(parts))

def main():
    torch.set_num_threads(2);out=ROOT/'results/architecture_audit_v1';out.mkdir(parents=True,exist_ok=True);rows=[]
    for generator,layers in [('ministral3_8b',34),('qwen3_8b',36)]:
        for name in POOLED+RAW:
            net=make_probe(name,(layers,96,4096));rows.append(dict(generator=generator,probe=name,parameters=sum(p.numel() for p in net.parameters()),trainable_parameters=sum(p.numel() for p in net.parameters() if p.requires_grad),layers=layers,tokens=96,hidden_dim=4096));del net
    with (out/'parameter_counts.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    diagnostics=[];torch.manual_seed(42);x=torch.randn(2,4,6,32)
    for name in ['M0','M1','M4']:
        net=make_probe(name,(4,6,32)).eval();a=x.mean(2) if name in POOLED else x
        with torch.inference_mode():delta=float((net(a)-net(a[:,[3,1,0,2]])).abs().max())
        if delta>1e-5:raise ValueError(f'Unexpected layer-order dependence: {name}:{delta}')
        diagnostics.append(dict(probe=name,max_absolute_logit_difference=delta,scope='Tiny structural sanity check, not empirical performance ablation'))
    atomic_json(out/'layer_permutation_diagnostic.json',diagnostics)
    diagram(out/'probe_architectures.svg','Probe computations verified from implementation',[
        ['M0: hidden states\nB × L × T × D','Mean over layers\nand tokens','Global feature\nB × D','MLP classifier','Hallucination\nprobability'],
        ['M1: hidden states\nB × L × T × D','Token mean\nShared layer projection','Content-based\nlayer weighting','MLP classifier','Hallucination\nprobability'],
        ['M2: hidden states\nB × L × T × D','Flatten layer × token\n2D positional encoding','Full attention\nCLS aggregation','Classifier head','Hallucination\nprobability'],
        ['M3: hidden states\nB × L × T × D','Token Transformer\nwithin each layer','Layer Transformer\nwith positions','Classifier head','Hallucination\nprobability'],
        ['M4: hidden states\nB × L × T × D','Multi-query layer fusion\nMean residual + gate','Token Transformer\nwith positions','Classifier head','Hallucination\nprobability']])
    diagram(out/'portability_concepts.svg','Three distinct portability questions',[
        ['Distribution','Same generator','Frozen probe','New dataset','Measure reliability'],
        ['Parameters','Updated generator','Same learned weights','Target evaluation','Measure transfer gap'],
        ['Architecture','Different generator','Same probe design','Fresh target training','Measure reusable design']])
    (out/'README.md').write_text('Parameter counts instantiate actual modern definitions, all trainable probe parameters included. M1 uses shared content-based layer scoring without depth positions; M4 also lacks layer-order positions. Therefore neither should be described as explicitly encoding layer identity. M0/M1/M4 layer permutation sanity checks confirm implemented invariance on a tiny tensor; this is not a trained performance ablation. M2/M3 have explicit layer positional encodings. Full scientific claims still require matched-capacity training ablations. Figures are editable vector SVG.\n')
    print(json.dumps(dict(parameter_rows=len(rows),diagnostics=diagnostics,output=str(out))),flush=True)
if __name__=='__main__':main()
