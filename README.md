# Service Matcher

A Python project exploring different approaches to matching equivalent services across two catalogues when the same service may be described using different terminology.

I compared three approaches — simple word overlap, TF-IDF, and pretrained sentence embeddings — against the same synthetic benchmark to understand how different text representations affect matching performance.

## The Problem

Equivalent services do not necessarily have similar names. For example:

```text
Flu Vaccination ↔ Influenza Immunisation
Blood Pressure Check ↔ Hypertension Assessment
```

A simple word-matching system struggles with these examples because there may be little or no vocabulary overlap.

The aim of this project was to rank provider services for each target service and evaluate whether the intended match was ranked highly.

## Methods

### 1. Word Overlap

A simple baseline written in Python. Service names are lowercased and split into sets of words, with similarity determined by the number of shared words.

This is easy to interpret but cannot recognise synonyms or related terminology.

### 2. TF-IDF + Cosine Similarity

Service names are converted into TF-IDF vectors using scikit-learn. TF-IDF weights words according to how distinctive they are across the catalogue, and cosine similarity is used to compare the resulting vectors.

This provides a more informative lexical representation, but still relies heavily on shared vocabulary.

### 3. Sentence Embeddings + Cosine Similarity

The final approach uses the pretrained `all-MiniLM-L6-v2` Sentence Transformer model to convert each service name into a 384-dimensional semantic embedding.

Cosine similarity is then used to rank provider services by the similarity of their embeddings. Unlike the lexical methods, this can identify relationships between differently worded phrases.

## Evaluation

The methods were evaluated on a synthetic dataset containing:

- 20 target services
- 35 provider services, including plausible distractors
- A manually defined ground-truth provider match for each target

### Results

| Method | Top-1 Correct | Top-1 Accuracy |
| --- | ---: | ---: |
| Word overlap | 5 / 20 | 25% |
| TF-IDF + cosine similarity | 4 / 20 | 20% |
| Sentence embeddings + cosine similarity | **12 / 20** | **60%** |

For the embedding method, the correct service appeared within the **top three candidates for all 20 targets**, giving **100% Top-3 accuracy on this synthetic benchmark**.

## Key Findings

Sentence embeddings substantially improved matching when equivalent services used different terminology. However, error analysis revealed an important limitation: **semantic similarity is not the same as service equivalence**.

For example, the embedding model ranked:

```text
Asthma Review
```

above the intended match:

```text
Asthma Medicines Consultation
```

for the target `Asthma Medication Review`, despite producing a high similarity score.

Similar errors occurred when small but important qualifiers such as *medication*, *annual*, *initial*, *childhood*, or *travel vaccination* distinguished related services.

The Top-3 results suggest that a more appropriate use of this type of matcher could be to **shortlist candidate services for human review**, rather than automatically accepting the highest-scoring match.

## Limitations

This is a small educational experiment using synthetic service names and manually defined ground truth. The reported results should therefore not be interpreted as estimates of real-world healthcare matching accuracy.

A production system would require a substantially larger validated dataset, more rigorous evaluation, and potentially additional information beyond service names.

## Running the Project

Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
python -m pip install -r requirements.txt
```

Run the evaluation:

```bash
python main.py
```

Run the automated test:

```bash
python -m unittest test_evaluation.py
```

The Sentence Transformer model may be downloaded automatically the first time the project is run.

## AI-Assisted Development

I built the initial matching algorithms and evaluation pipeline myself, then used OpenAI Codex for repository-level code review and targeted refactoring.

Codex helped identify a fragile assumption that depended on the ground-truth file having the same row ordering as the target catalogue. I reviewed and corrected this using ID-based lookups. I also used Codex to help reduce duplicated evaluation logic and add an automated test verifying that reordering the answer key does not change the results.

Git was used throughout the project to review changes and maintain working checkpoints.