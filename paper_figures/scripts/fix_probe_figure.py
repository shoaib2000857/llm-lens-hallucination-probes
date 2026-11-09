from pathlib import Path
p = Path(__file__).resolve().parent.parent / 'sections' / '05_probes.tex'
L = p.read_text(encoding='utf-8').split('\n')
start = next(i for i, l in enumerate(L) if l.startswith(r'\begin{tikzpicture}'))
end = next(i for i, l in enumerate(L) if l.startswith(r'\end{tikzpicture}'))
rows = [('M0', r'mean over $L$ and $T$', r'two-layer MLP'),
        ('M1', r'mean over $T$ within each layer', r'shared projection; content-scored softmax over layers'),
        ('M2', r'project each cell; flatten the $L\times T$ grid', r'2-D sinusoidal positions; transformer encoder'),
        ('M3', r'temporal transformer within each layer', r'transformer across layer summaries (layer positions)'),
        ('M4', r'learned-query fusion over layers at each step', r'temporal transformer (token positions)')]
new = [r'\begin{tikzpicture}[font=\footnotesize,',
       r'  inp/.style={draw=cInk, rounded corners=2pt, fill=cBox, minimum height=0.62cm, minimum width=1.0cm, align=center},',
       r'  op/.style={draw=cInk, rounded corners=2pt, fill=white, minimum height=0.62cm, text width=#1, align=center},',
       r'  prob/.style={draw=cQwen, thick, rounded corners=2pt, fill=white, minimum height=0.62cm, minimum width=0.7cm},',
       r'  arr/.style={-{Stealth[length=4pt]}, cInk, thick}]']
for k, (m, a, b) in enumerate(rows):
    y = f'{-1.0 * k:.2f}'
    new += [rf'\node[font=\footnotesize\bfseries, anchor=east] at (-0.15,{y}) {{{m}}};',
            rf'\node[inp] (a{k}) at (0.55,{y}) {{$\Hs$}};',
            rf'\node[op=4.4cm] (b{k}) at (3.95,{y}) {{{a}}};',
            rf'\node[op=5.4cm] (c{k}) at (9.35,{y}) {{{b}}};',
            rf'\node[prob] (d{k}) at (12.65,{y}) {{$\hat p$}};',
            rf'\draw[arr] (a{k})--(b{k}); \draw[arr] (b{k})--(c{k}); \draw[arr] (c{k})--(d{k});']
new.append(r'\end{tikzpicture}')
p.write_text('\n'.join(L[:start] + new + L[end + 1:]), encoding='utf-8')
print('ok')
