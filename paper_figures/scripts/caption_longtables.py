"""Write captioned copies of the EMNLP longtables for Paper A (originals untouched)."""
from pathlib import Path
import re
T = Path(__file__).resolve().parent.parent / 'tables'
specs = {
    'H4_probe_vs_human': ('All eight frozen probes and the best-layer control against resolved human references, by dataset and generator. Matched judge AUROC uses consensus labels on the same responses. Mean per-seed values; 95\\% context-cluster intervals.', 'tab:humanfull'),
    'update_complete': ('Every LoRA checkpoint: frozen source probe, scratch and source-initialized fits with 500 target labels, and full-budget scratch reference (mean per-seed \\auroc{}).', 'tab:updates'),
}
for name, (cap, lab) in specs.items():
    src = (T / 'emnlp' / f'{name}.tex').read_text(encoding='utf-8')
    m = re.match(r'(\\begin\{longtable\}\{[^}]*\}\n)', src)
    out = m.group(1) + '\\caption{' + cap + '}\\label{' + lab + '}\\\\\n' + src[m.end():]
    (T / f'{name}_cap.tex').write_text(out, encoding='utf-8')
    print('wrote', name)
