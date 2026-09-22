# Data

The repository does not redistribute SciFact.

The experiments use the BEIR version of **SciFact** with:

- 5,183 scientific abstracts in the corpus
- 300 test claims with relevance judgements
- corpus fields for document ID, title, and abstract text
- query files containing scientific claims
- qrels files linking claims to relevant evidence documents

The original notebooks download the dataset from the public BEIR SciFact archive. The cleaned code expects the standard BEIR/SciFact JSONL and TSV layout.

The project works with abstracts, not full scientific papers.
