# Legacy tables extracted verbatim (for provenance review; not re-verified numbers)

## V1_arxiv_2604.06277

Caption: 5-fold cross-validation performance (mean $\pm$ std) across the five trained architectures (M0--M4). All probes are trained on $y_{\text{judge}}$ labels. Higher AUC/F1/Acc and lower ECE are better.

```latex
\begin{table}[htbp]
		\centering
		\begin{tabular}{lcccc}
			\toprule
			Model & AUC & F1 & ECE & Acc \\
			\midrule
			M0: ProbeMLP & 0.7560 $\pm$ 0.0079 & 0.4969 $\pm$ 0.0069 & 0.2370 $\pm$ 0.0583 & 0.6687 \\
			M1: LayerWiseMLP & 0.7999 $\pm$ 0.0105 & 0.5403 $\pm$ 0.0103 & 0.2258 $\pm$ 0.0434 & 0.7064 \\
			M2: CrossLayerTransformer & 0.8515 $\pm$ 0.0054 & 0.6096 $\pm$ 0.0154 & 0.1721 $\pm$ 0.0600 & 0.7851 \\
			M3: HierarchicalTransformer & 0.8490 $\pm$ 0.0089 & 0.5898 $\pm$ 0.0219 & 0.1845 $\pm$ 0.0791 & 0.7689 \\
			M4: CrossLayerAttentionTransformerV2 & 0.8275 $\pm$ 0.0066 & 0.5845 $\pm$ 0.0142 & 0.1231 $\pm$ 0.0461 & 0.7920 \\
			\bottomrule
		\end{tabular}
		\caption{5-fold cross-validation performance (mean $\pm$ std) across the five trained architectures (M0--M4). All probes are trained on $y_{\text{judge}}$ labels. Higher AUC/F1/Acc and lower ECE are better.}
		\label{tab:main_results}
	\end{table}
```

## V1_arxiv_2604.06277

Caption: Single-fold validation metrics from best-checkpoint analysis. Here, $\tau^*$ denotes the threshold maximizing validation F1.

```latex
\begin{table}[htbp]
		\centering
		\begin{tabular}{lcccccc}
			\toprule
			Model & AUC & F1@0.5 & Best F1 & $\tau^*$ & ECE & Acc \\
			\midrule
			M0: ProbeMLP & 0.7271 & 0.4864 & 0.4864 & 0.50 & 0.2468 & 0.6590 \\
			M1: LayerWiseMLP & 0.7921 & 0.5463 & 0.5463 & 0.50 & 0.1739 & 0.7343 \\
			M2: CrossLayerTransformer & 0.8535 & 0.6250 & 0.6381 & 0.75 & 0.1787 & 0.7943 \\
			M3: HierarchicalTransformer & 0.8577 & 0.6404 & 0.6445 & 0.65 & 0.1051 & 0.8219 \\
			M4: CrossLayerAttentionTransformerV2 & 0.8377 & 0.6098 & 0.6099 & 0.55 & 0.1308 & 0.8038 \\
			\bottomrule
		\end{tabular}
		\caption{Single-fold validation metrics from best-checkpoint analysis. Here, $\tau^*$ denotes the threshold maximizing validation F1.}
		\label{tab:single_fold_results}
	\end{table}
```

## V1_arxiv_2604.06277

Caption: Final held-out test-set results (5,000 rows) on the held-out test dataset. Here, $\tau^*$ is the threshold that maximizes F1 on the evaluation sweep.

```latex
\begin{table}[htbp]
		\centering
		\begin{tabular}{lccccccc}
			\toprule
			Model & AUC & PR-AUC & F1 & Acc & ECE & Best F1 & $\tau^*$ \\
			\midrule
			M0: ProbeMLP & 0.7441 & 0.5124 & 0.5316 & 0.6712 & 0.2092 & 0.5351 & 0.5468 \\
			M1: LayerWiseMLP & 0.8074 & 0.6070 & 0.6025 & 0.7514 & 0.1430 & 0.6084 & 0.4581 \\
			M2: CrossLayerTransformer & 0.8490 & 0.6960 & 0.6475 & 0.7742 & 0.1747 & 0.6600 & 0.6453 \\
			M3: HierarchicalTransformer & 0.8577 & 0.7057 & 0.6644 & 0.8040 & 0.1025 & 0.6653 & 0.4729 \\
			M4: CrossLayerAttentionTransformerV2 & 0.8294 & 0.6454 & 0.6236 & 0.7822 & 0.1136 & 0.6282 & 0.3646 \\
			\bottomrule
		\end{tabular}
		\caption{Final held-out test-set results (5,000 rows) on the held-out test dataset. Here, $\tau^*$ is the threshold that maximizes F1 on the evaluation sweep.}
		\label{tab:test_results}
	\end{table}
```

## V1_arxiv_2604.06277

Caption: Probe-only benchmark with hidden states precomputed. ``Batch lat.'' denotes amortized latency under batched inference; ``Single lat.'' denotes one-sample latency.

```latex
\begin{table}[htbp]
		\centering
		\small
		\begin{tabular}{lcccc}
			\toprule
			Model & Batch lat. (ms) & Single lat. (ms) & Samples/s & GPU mem (MB) \\
			\midrule
			M0: ProbeMLP & 0.15 & 1.66 & 6687.96 & 584.51 \\
			M1: LayerWiseMLP & 0.19 & 1.55 & 5362.70 & 589.26 \\
			M2: CrossLayerTransformer & 5.62 & 6.66 & 177.83 & 629.25 \\
			M3: HierarchicalTransformer & 2.59 & 4.02 & 385.43 & 628.68 \\
			M4: CrossLayerAttentionTransformerV2 & 1.34 & 4.68 & 745.58 & 603.66 \\
			\bottomrule
		\end{tabular}
		\caption{Probe-only benchmark with hidden states precomputed. ``Batch lat.'' denotes amortized latency under batched inference; ``Single lat.'' denotes one-sample latency.}
		\label{tab:probe_only_efficiency}
	\end{table}
```

## V1_arxiv_2604.06277

Caption: End-to-end deployment-style benchmark including generation, hidden-state extraction, and probe scoring. ``Avg prob.'' is the mean predicted hallucination probability over the benchmark set.

```latex
\begin{table}[htbp]
		\centering
		\small
		\begin{tabular}{lccccc}
			\toprule
			Model & Gen. (s) & Probe (ms) & Total (s) & Total q/s & Avg prob. \\
			\midrule
			M0: ProbeMLP & 4.3313 & 1.262 & 4.3325 & 0.231 & 0.549 \\
			M1: LayerWiseMLP & 4.3313 & 1.584 & 4.3329 & 0.231 & 0.489 \\
			M2: CrossLayerTransformer & 4.3313 & 6.709 & 4.3380 & 0.231 & 0.470 \\
			M3: HierarchicalTransformer & 4.3313 & 3.637 & 4.3349 & 0.231 & 0.456 \\
			M4: CrossLayerAttentionTransformerV2 & 4.3313 & 4.758 & 4.3360 & 0.231 & 0.332 \\
			\bottomrule
		\end{tabular}
		\caption{End-to-end deployment-style benchmark including generation, hidden-state extraction, and probe scoring. ``Avg prob.'' is the mean predicted hallucination probability over the benchmark set.}
		\label{tab:e2e_efficiency}
	\end{table}
```

## V1_arxiv_2604.06277

Caption: Output-level similarity baselines and best probe result on the held-out 5,000-row test set.

```latex
\begin{table}[htbp]
		\centering
		\begin{tabular}{lccc}
			\toprule
			Method & AUC & F1 & Acc \\
			\midrule
			Cosine Similarity (Max) & 0.4110 & 0.4269 & 0.5146 \\
			Mean Similarity & 0.4092 & 0.4220 & 0.4900 \\
			Softmax Similarity & 0.4105 & 0.4254 & 0.5068 \\
			Token F1 Similarity & 0.4222 & 0.3652 & 0.3116 \\
			\midrule
			Best Probe (M3: HierarchicalTransformer) & 0.8577 & 0.7057 & 0.6644 \\
			\bottomrule
		\end{tabular}
		\caption{Output-level similarity baselines and best probe result on the held-out 5,000-row test set.}
		\label{tab:similarity_baselines}
	\end{table}
```

## V1_arxiv_2604.06277

Caption: Diagnostic metrics computed on the full training-data evaluation run (\texttt{eval 10}, 10k samples). These are not held-out test metrics and should not be interpreted as final generalization performance.

```latex
\begin{table}[htbp]
		\centering
		\begin{tabular}{lccccc}
			\toprule
			Model ID & Acc & Best F1 & ROC-AUC & PR-AUC & ECE \\
			\midrule
			M0 & 0.8925 & 0.8262 & 0.9528 & 0.8880 & 0.1145 \\
			M1 & 0.8127 & 0.6357 & 0.8655 & 0.6760 & 0.1515 \\
			M2 & 0.8641 & 0.7520 & 0.9255 & 0.8123 & 0.1754 \\
			M3 & 0.7472 & 0.6959 & 0.8954 & 0.7520 & 0.2484 \\
			M4 & 0.8702 & 0.7396 & 0.9187 & 0.7976 & 0.1265 \\
			\bottomrule
		\end{tabular}
		\caption{Diagnostic metrics computed on the full training-data evaluation run (\texttt{eval 10}, 10k samples). These are not held-out test metrics and should not be interpreted as final generalization performance.}
		\label{tab:full_dataset_diagnostics}
	\end{table}
```

## ARR2721

Caption: Cross-dataset AUC and F1@0.5 (LLaMA probes on LLaMA-2-7B, judge labels, $n{=}1{,}000$ per dataset). Bold = best individual probe per row. Comparing AUC and F1 reveals the class-imbalance effect: TriviaQA's AUC of 0.73--0.80 collapses to F1 of 0.22--0.35 at a fixed 0.5 threshold due to only 6\% positive rate, confirming threshold calibration is essential for deployment.

```latex
\begin{table*}[t]
\centering
\setlength{\tabcolsep}{4.5pt}
\resizebox{\textwidth}{!}{%
\begin{tabular}{l cccccc cccccc}
\toprule
 & \multicolumn{6}{c}{\textbf{AUC-ROC}} & \multicolumn{6}{c}{\textbf{F1@0.5}} \\
\cmidrule(lr){2-7} \cmidrule(lr){8-13}
Dataset & M0 & M1 & M2 & M3 & M4 & Ens. & M0 & M1 & M2 & M3 & M4 & Ens. \\
\midrule
SQuAD v2  & 0.752 & 0.787 & \textbf{0.851} & 0.846 & 0.821 & 0.846 & 0.489 & 0.500 & \textbf{0.599} & 0.509 & 0.526 & 0.558 \\
NQ-Open   & 0.714 & 0.724 & \textbf{0.804} & 0.791 & 0.787 & 0.803 & 0.393 & 0.339 & \textbf{0.503} & 0.449 & 0.458 & 0.508 \\
TriviaQA  & 0.733 & 0.709 & 0.780 & \textbf{0.798} & 0.789 & 0.805 & 0.249 & 0.222 & \textbf{0.348} & 0.241 & 0.349 & 0.333 \\
HotpotQA  & 0.598 & 0.675 & \textbf{0.728} & 0.727 & 0.660 & 0.704 & 0.539 & 0.553 & \textbf{0.554} & 0.530 & 0.532 & 0.548 \\
PubMedQA  & 0.644 & \textbf{0.672} & 0.659 & 0.644 & 0.650 & 0.682 & 0.336 & 0.367 & 0.263 & \textbf{0.379} & 0.350 & 0.364 \\
\midrule
\textbf{Mean} & 0.688 & 0.713 & \textbf{0.764} & 0.761 & 0.741 & 0.768 & 0.401 & 0.396 & \textbf{0.453} & 0.422 & 0.443 & 0.462 \\
\bottomrule
\end{tabular}%
}
\caption{Cross-dataset AUC and F1@0.5 (LLaMA probes on LLaMA-2-7B, judge labels, $n{=}1{,}000$ per dataset). Bold = best individual probe per row. Comparing AUC and F1 reveals the class-imbalance effect: TriviaQA's AUC of 0.73--0.80 collapses to F1 of 0.22--0.35 at a fixed 0.5 threshold due to only 6\% positive rate, confirming threshold calibration is essential for deployment.}
\label{tab:cross_dataset}
\end{table*}
```

## ARR2721

Caption: Label statistics by dataset (LLaMA generator, $n{=}1{,}000$). Gold--judge agreement correlates with cross-dataset probe reliability. TriviaQA's high agreement reflects clean grounding decisions; SQuAD v2 and PubMedQA's low agreement reflects inherently harder cases.

```latex
\begin{table}[t]
\centering
\resizebox{\columnwidth}{!}{%
\begin{tabular}{lccc}
\toprule
Dataset & Judge pos. & Gold pos. & Agree. \\
\midrule
TriviaQA  & 6.0\%  & 3.3\%  & 92.7\% \\
NQ-Open   & 18.2\% & 18.2\% & 79.2\% \\
HotpotQA  & 33.5\% & 56.6\% & 58.9\% \\
PubMedQA  & 19.3\% & 44.5\% & 54.6\% \\
SQuAD v2  & 21.3\% & 66.6\% & 51.1\% \\
\bottomrule
\end{tabular}%
}
\caption{Label statistics by dataset (LLaMA generator, $n{=}1{,}000$). Gold--judge agreement correlates with cross-dataset probe reliability. TriviaQA's high agreement reflects clean grounding decisions; SQuAD v2 and PubMedQA's low agreement reflects inherently harder cases.}
\label{tab:agreement}
\end{table}
```

## ARR2721

Caption: Same-family model transfer (ensemble, judge labels, 100 samples per dataset, 5 datasets). ``Avg AUC'' = mean best-mode AUC; ``Inv./5'' = number of datasets requiring score inversion for best discrimination.

```latex
\begin{table}[t]
\centering
\resizebox{\columnwidth}{!}{%
\begin{tabular}{llcc}
\toprule
Probe family & Target generator & Avg AUC & Inv./5 \\
\midrule
\multirow{5}{*}{LLaMA probes}
 & LLaMA-2-7B base        & 0.722 & 0 \\
 & LLaMA-2-7B chat        & 0.757 & 1 \\
 & LLaMA-3-8B base        & 0.581 & 4 \\
 & LLaMA-3.1-8B base      & 0.578 & 4 \\
 & LLaMA-3.1-8B Instruct  & 0.594 & 2 \\
\midrule
\multirow{4}{*}{Mistral probes}
 & Mistral-7B v0.1 base       & 0.663 & 0 \\
 & Mistral-7B Instruct v0.1   & 0.733 & 0 \\
 & Mistral-7B Instruct v0.2   & \textbf{0.816} & 1 \\
 & Mistral-7B v0.3 base       & 0.618 & 1 \\
\bottomrule
\end{tabular}%
}
\caption{Same-family model transfer (ensemble, judge labels, 100 samples per dataset, 5 datasets). ``Avg AUC'' = mean best-mode AUC; ``Inv./5'' = number of datasets requiring score inversion for best discrimination.}
\label{tab:family_sweep}
\end{table}
```

## ARR2721

Caption: Cross-family transfer summary (ensemble, judge labels, $n{=}1{,}000$, averaged over 5 datasets). AUC$^\dagger$ = inverted-score AUC. LLaMA$\to$Mistral fires near-universally (mean prob 0.81) with partial ranking preserved; Mistral$\to$LLaMA shows partial sign reversal (AUC $<$ 0.5).

```latex
\begin{table}[t]
\centering
\resizebox{\columnwidth}{!}{%
\begin{tabular}{lcccc}
\toprule
Direction & Avg AUC & Avg AUC$^\dagger$ & Mean prob & Mode \\
\midrule
LLaMA $\to$ Mistral & 0.544 & 0.456 & 0.810 & Drift \\
Mistral $\to$ LLaMA & 0.427 & 0.573 & 0.732 & Inversion \\
\bottomrule
\end{tabular}%
}
\caption{Cross-family transfer summary (ensemble, judge labels, $n{=}1{,}000$, averaged over 5 datasets). AUC$^\dagger$ = inverted-score AUC. LLaMA$\to$Mistral fires near-universally (mean prob 0.81) with partial ranking preserved; Mistral$\to$LLaMA shows partial sign reversal (AUC $<$ 0.5).}
\label{tab:cross_family_summary}
\end{table}
```

## ARR2721

Caption: Cross-family transfer by dataset (ensemble). LLaMA$\to$Mistral: AUC consistently above 0.5 but probabilities severely inflated. Mistral$\to$LLaMA: AUC consistently below 0.5, confirming systematic sign reversal across all datasets. TriviaQA and NQ-Open are most inverted ($\leq$0.37).

```latex
\begin{table}[t]
\centering
\resizebox{\columnwidth}{!}{%
\begin{tabular}{lcc|cc}
\toprule
 & \multicolumn{2}{c|}{LLaMA$\to$Mistral} & \multicolumn{2}{c}{Mistral$\to$LLaMA} \\
Dataset & AUC & Mean prob & AUC & Mean prob \\
\midrule
SQuAD v2  & 0.567 & 0.821 & 0.458 & 0.730 \\
NQ-Open   & 0.535 & 0.793 & 0.364 & 0.661 \\
TriviaQA  & 0.453 & 0.811 & 0.367 & 0.763 \\
HotpotQA  & 0.595 & 0.761 & 0.474 & 0.753 \\
PubMedQA  & 0.568 & 0.863 & 0.471 & 0.753 \\
\midrule
Mean      & 0.544 & 0.810 & 0.427 & 0.732 \\
\bottomrule
\end{tabular}%
}
\caption{Cross-family transfer by dataset (ensemble). LLaMA$\to$Mistral: AUC consistently above 0.5 but probabilities severely inflated. Mistral$\to$LLaMA: AUC consistently below 0.5, confirming systematic sign reversal across all datasets. TriviaQA and NQ-Open are most inverted ($\leq$0.37).}
\label{tab:cross_family_dataset}
\end{table}
```

## ARR2721

Caption: Cross-family per-probe AUC (avg.\ over 5 datasets, $n{=}1{,}000$ each). AUC$^\dagger$ = inverted-score AUC. MProb = mean predicted hallucination probability. Three patterns emerge: M1 is least inverted in Mistral$\to$LLaMA (0.478); M3's high MProb in LLaMA$\to$Mistral (0.92) indicates near-universal overfiring; M4 is most extreme in both directions.

```latex
\begin{table}[t]
\centering
\resizebox{\columnwidth}{!}{%
\begin{tabular}{lccc|ccc}
\toprule
 & \multicolumn{3}{c|}{LLaMA$\to$Mistral} & \multicolumn{3}{c}{Mistral$\to$LLaMA} \\
Probe & AUC & AUC$^\dagger$ & MProb & AUC & AUC$^\dagger$ & MProb \\
\midrule
M0   & 0.540 & 0.460 & 0.86 & 0.431 & 0.569 & 0.77 \\
M1   & 0.552 & 0.448 & 0.60 & 0.478 & 0.522 & 0.72 \\
M2   & 0.510 & 0.490 & 0.85 & 0.420 & 0.580 & 0.72 \\
M3   & 0.521 & 0.479 & 0.92 & 0.481 & 0.519 & 0.94 \\
M4   & 0.532 & 0.468 & 0.72 & 0.442 & 0.558 & 0.54 \\
Ens. & 0.544 & 0.456 & 0.81 & 0.427 & 0.573 & 0.73 \\
\bottomrule
\end{tabular}%
}
\caption{Cross-family per-probe AUC (avg.\ over 5 datasets, $n{=}1{,}000$ each). AUC$^\dagger$ = inverted-score AUC. MProb = mean predicted hallucination probability. Three patterns emerge: M1 is least inverted in Mistral$\to$LLaMA (0.478); M3's high MProb in LLaMA$\to$Mistral (0.92) indicates near-universal overfiring; M4 is most extreme in both directions.}
\label{tab:cross_family_probes}
\end{table}
```

## ARR2721

Caption: Full per-probe AUC and PR-AUC (LLaMA probes on LLaMA-2-7B, $n{=}1{,}000$ per dataset, judge labels).

```latex
\begin{table}[h]
	\centering
	\small
	\setlength{\tabcolsep}{4pt}
	\begin{tabular}{llcc}
		\toprule
		Dataset & Probe & AUC & PR-AUC \\
		\midrule
		\multirow{5}{*}{SQuAD v2}  
		& M0 & 0.752 & 0.427 \\ 
		& M1 & 0.787 & 0.514 \\ 
		& M2 & 0.851 & 0.651 \\ 
		& M3 & 0.846 & 0.648 \\ 
		& M4 & 0.821 & 0.582 \\
		\midrule
		\multirow{5}{*}{NQ-Open}   
		& M0 & 0.714 & 0.288 \\ 
		& M1 & 0.724 & 0.301 \\ 
		& M2 & 0.804 & 0.523 \\ 
		& M3 & 0.791 & 0.522 \\ 
		& M4 & 0.787 & 0.476 \\
		\midrule
		\multirow{5}{*}{TriviaQA}  
		& M0 & 0.733 & 0.197 \\ 
		& M1 & 0.709 & 0.149 \\ 
		& M2 & 0.780 & 0.400 \\ 
		& M3 & 0.798 & 0.376 \\ 
		& M4 & 0.789 & 0.297 \\
		\midrule
		\multirow{5}{*}{HotpotQA}  
		& M0 & 0.598 & 0.392 \\ 
		& M1 & 0.675 & 0.473 \\ 
		& M2 & 0.728 & 0.590 \\ 
		& M3 & 0.727 & 0.579 \\ 
		& M4 & 0.660 & 0.505 \\
		\midrule
		\multirow{5}{*}{PubMedQA}  
		& M0 & 0.644 & 0.320 \\ 
		& M1 & 0.672 & 0.369 \\ 
		& M2 & 0.659 & 0.308 \\ 
		& M3 & 0.644 & 0.278 \\ 
		& M4 & 0.650 & 0.322 \\
		\bottomrule
	\end{tabular}
	\caption{Full per-probe AUC and PR-AUC (LLaMA probes on LLaMA-2-7B, $n{=}1{,}000$ per dataset, judge labels).}
	\label{tab:full_pr_auc}
\end{table}
```

## ARR2721

Caption: Source model performance: LLaMA probes on the 5,000-row held-out SQuAD v2 test set. $\tau^*$ = optimal F1 threshold.

```latex
\begin{table}[!h]
	\centering
	\small
	\setlength{\tabcolsep}{3pt}
	\begin{tabular}{lcccccc}
		\toprule
		Probe & AUC & PR-AUC & F1 & Best F1 & $\tau^*$ & ECE \\
		\midrule
		M0 & 0.744 & 0.512 & 0.532 & 0.535 & 0.547 & 0.209 \\
		M1 & 0.807 & 0.607 & 0.603 & 0.608 & 0.458 & 0.143 \\
		M2 & 0.849 & 0.696 & 0.648 & 0.660 & 0.645 & 0.175 \\
		M3 & 0.858 & 0.706 & 0.664 & 0.665 & 0.473 & 0.103 \\
		M4 & 0.829 & 0.645 & 0.624 & 0.628 & 0.365 & 0.114 \\
		\bottomrule
	\end{tabular}
	\caption{Source model performance: LLaMA probes on the 5,000-row held-out SQuAD v2 test set. $\tau^*$ = optimal F1 threshold.}
	\label{tab:source_perf}
\end{table}
```
