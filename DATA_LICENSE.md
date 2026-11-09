# Data licences

| Path | Contents | Licence |
|---|---|---|
| `data/labels/` | Judge verdicts, consensus labels, token-confidence summaries | CC BY 4.0 |
| `data/human/` | De-identified human reference labels and agreement tables | CC BY 4.0 |
| `data/predictions/` | Probe output probabilities for every reported fit | CC BY 4.0 |
| `results/` | Derived metric tables | CC BY 4.0 |
| `data/responses/` | Questions and gold answers from the source datasets, plus model-generated answers | **CC BY-SA 4.0** (share-alike, inherited from SQuAD 2.0, HotpotQA and Natural Questions) |

Source passages are **not** redistributed. Each response record carries `original_id` and `context_sha256`, so the passage can be recovered from the original dataset release.

Upstream terms continue to apply:
- **Datasets:** SQuAD 2.0 (CC BY-SA 4.0), HotpotQA (CC BY-SA 4.0), Natural Questions (CC BY-SA 3.0), PubMedQA (MIT), TriviaQA (see the dataset release).
- **Generator and judge models:** generated text was produced with Qwen3-8B, Qwen3-8B-Base, Ministral-3-8B (Base/Instruct-2512), Mistral-7B-Instruct-v0.2 and Phi-4-mini-instruct. See each model card for the terms that apply to outputs.
