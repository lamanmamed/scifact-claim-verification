# SciFact Claim Verification

A scientific claim verification system built on the SciFact dataset. Given a claim, the system retrieves relevant scientific abstracts, improves their ranking, extracts focused evidence, and predicts one of three labels:

- **SUPPORTS**
- **REFUTES**
- **NOT ENOUGH INFO**

## What the system does

The final pipeline is:

```text
scientific claim
      ↓
rewrite into a sparse query and a dense query
      ↓
BM25 retrieval + MPNet dense retrieval
      ↓
min-max score normalization and hybrid fusion
      ↓
BGE cross-encoder reranking
      ↓
SpanBERT evidence extraction from the top 3 abstracts
      ↓
Qwen2.5-3B-Instruct
      ↓
SUPPORTS / REFUTES / NOT ENOUGH INFO
```

The retrieval stages are evaluated separately from the final verdict generator, which makes it possible to see where each component helps and where it does not.

## Data

The project uses **SciFact**, a dataset of scientific claims and evidence documents. The saved experiments use:

- **5,183 scientific abstracts**
- **300 test claims**
- relevance judgements connecting claims to supporting or refuting documents

Only abstracts are used, not full papers.

Before retrieval, each document is stored in three forms:

- the original title + abstract with whitespace cleaned
- a dense-retrieval version that keeps the scientific text unchanged
- a BM25 version that lowercases text and normalizes a few dash characters

Scientific symbols and other Unicode characters are kept rather than stripped out.

## Dense retrieval

Several sentence encoders were compared before selecting MPNet. The separate dense-retriever benchmark recorded **Recall@20 = 0.8530** for the original `all-mpnet-base-v2` model.

MPNet was then fine-tuned on SciFact claim-document pairs with `MultipleNegativesRankingLoss`. In that benchmark run, Recall@20 increased to **0.9109**.

| Dense MPNet run | Recall@20 | nDCG@20 | MRR |
| --- | ---: | ---: | ---: |
| Original MPNet | 0.8530 | 0.6700 | 0.6185 |
| Fine-tuned MPNet | **0.9109** | **0.7261** | **0.6749** |

The dense-retriever benchmark and the later end-to-end pipeline are separate saved runs. The final pipeline notebook records a different MPNet test result, so the two sets of numbers are kept separate rather than mixed together.

## Combining BM25 and dense retrieval

For each claim, BM25 and dense retrieval each return candidate documents. Their scores are min-max normalized independently and combined with equal weight:

```text
hybrid_score = 0.5 × normalized_BM25 + 0.5 × normalized_dense
```

The final retrieval comparison on 300 test claims was:

| Retrieval method | Recall@20 | nDCG@20 |
| --- | ---: | ---: |
| BM25 | 0.8521 | 0.6919 |
| Dense MPNet | 0.8946 | 0.7186 |
| Dense MiniLM | 0.8373 | 0.6630 |
| Hybrid MPNet | **0.9279** | **0.7600** |
| Hybrid MiniLM | 0.8767 | 0.7319 |

Hybrid MPNet therefore became the retrieval method used by the final system.

## Rewriting a claim before retrieval

The system can turn one scientific claim into two different queries:

- a short **sparse query** for BM25
- a full-sentence **dense query** for the embedding model

The prompts explicitly tell the rewriter not to change causality, direction, negation, or uncertainty. For example, `does not prevent` should not become `prevents`, and `may reduce` should not become `reduces`.

The saved final implementation compares a controlled Qwen rewrite method with a more detailed Llama rewrite method.

| Hybrid MPNet query | Recall@20 | nDCG@20 | MRR |
| --- | ---: | ---: | ---: |
| Original claim | 0.9279 | **0.7600** | **0.7139** |
| Qwen rewrite | **0.9336** | 0.7561 | 0.7085 |
| Llama rewrite | 0.9286 | 0.7475 | 0.6960 |

The Qwen rewrite increased Recall@20 slightly, but it did **not** improve nDCG@20 or MRR. This is why the repository keeps query rewriting as its own component instead of treating rewriting as automatically beneficial.

## Reranking the retrieved documents

The next stage takes the retrieved candidates and scores each `(claim, document)` pair with a cross-encoder.

The BGE reranker was fine-tuned on SciFact with three types of training information:

- relevant documents as positives
- highly ranked but non-relevant documents as hard negatives
- randomly sampled non-relevant documents as easier negatives

For each positive/negative pair, training uses a pairwise logistic objective that rewards:

```text
score(claim, positive) > score(claim, negative)
```

On the Qwen-rewritten Hybrid MPNet results:

| Reranking | nDCG@3 | Recall@3 |
| --- | ---: | ---: |
| No reranker | 0.6955 | 0.7591 |
| MiniLM reranker | 0.6577 | 0.7094 |
| Fine-tuned BGE reranker | **0.7546** | **0.7988** |

The MiniLM reranker made the ranking worse in this run. The fine-tuned BGE model improved both top-three ranking quality and top-three recall, so it was used in the final pipeline.

## Generating the verdict

The top three reranked documents are passed to an instruction-tuned language model. The prompt asks the model to:

1. summarize each evidence item
2. choose `SUPPORTS`, `REFUTES`, or `NOT ENOUGH INFO`
3. explain the decision using only the supplied evidence
4. identify contradictory evidence when the verdict is `REFUTES`

The first generator comparison was:

| Generator | Overall accuracy | SUPPORTS | REFUTES | NOT ENOUGH INFO |
| --- | ---: | ---: | ---: | ---: |
| Qwen2.5-3B-Instruct | **0.543** | 0.758 | 0.391 | 0.393 |
| Llama-3.2-3B-Instruct | 0.427 | **0.815** | **0.422** | 0.000 |

Llama performed reasonably on SUPPORTS and REFUTES but failed to identify NOT ENOUGH INFO in this evaluation. Qwen had better overall balance and was selected for the final system.

## Extracting shorter evidence with SpanBERT

Passing a full abstract to the generator can include a lot of text unrelated to the claim. The final version therefore runs a SpanBERT question-answering model over each of the top three abstracts and extracts the sentence region around the highest-scoring answer span.

The generator then sees those focused snippets instead of the longer abstracts.

| Evidence passed to Qwen | Overall accuracy | SUPPORTS | REFUTES | NOT ENOUGH INFO |
| --- | ---: | ---: | ---: | ---: |
| Full abstract text | 0.543 | **0.758** | 0.391 | 0.393 |
| SpanBERT snippets | **0.570** | 0.702 | **0.469** | **0.482** |

Span extraction raised overall verdict accuracy from **54.3% to 57.0%**. The gain came mainly from better REFUTES and NOT ENOUGH INFO predictions, while SUPPORTS accuracy decreased.

## Final configuration

The final saved configuration uses:

| Component | Choice |
| --- | --- |
| Retrieval | Hybrid BM25 + MPNet |
| Query rewriting | Controlled Qwen rewrite |
| Reranker | Fine-tuned BGE |
| Evidence extraction | SpanBERT |
| Verdict generator | Qwen2.5-3B-Instruct |
| Retrieved candidates | 20 |
| Evidence passed to generator | top 3 |

This separation is important because the strongest choice at one stage was not always the strongest at another. Query rewriting, for example, improved recall slightly but reduced ranking metrics, while BGE reranking gave a clear improvement at the top of the list.

## Repository structure

```text
scifact-claim-verification/
├── src/scifact_verification/
│   ├── data.py
│   ├── dense_training.py
│   ├── retrieval.py
│   ├── rewriting.py
│   ├── reranking.py
│   ├── evidence.py
│   ├── generation.py
│   ├── metrics.py
│   └── pipeline.py
├── experiments/
│   └── recorded_results.py
├── scripts/
│   └── show_results.py
├── tests/
├── data/
│   └── README.md
├── pyproject.toml
└── README.md
```

## Running the lightweight code

Install the package:

```bash
pip install -e .
```

Run the tests:

```bash
python -m unittest discover -s tests -v
```

Print the recorded retrieval, reranking, and generation results:

```bash
python scripts/show_results.py
```

The full model pipeline requires the optional retrieval and transformer dependencies plus the downloaded SciFact data and model checkpoints.

## Authentication

Some Hugging Face models require authentication. Supply credentials through the environment rather than placing tokens in source code:

```bash
export HF_TOKEN="..."
```

Model checkpoints, FAISS indexes, dataset files, credentials, and notebook outputs are deliberately excluded from the repository.