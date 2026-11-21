# Task #5: Quality Rubric and LLM Labeling

**Goal:** finalize the comment-quality rubric, build a reference set of
labeled comments, have an LLM label more comments with the rubric, and measure
how well the LLM agrees with the reference labels.

**Results so far:** Claude Opus 5.5 matches the reference labels on
civility 93% of the time (Cohen's kappa 0.87) and is never more than one point
off on any dimension. Reasoning is the weakest dimension (73%, kappa 0.61).
A first lightweight scorer trained on the LLM labels (section 7) runs in
seconds but is only modestly better than guessing the average. Relevance has
almost no variation and is recommended for dropping.

## 1. The rubric

Full rubric: [`quality-rubric.md`](quality-rubric.md). Each reply is scored 0,
1 or 2 on four dimensions:

| Dimension        | Question                             | Notes                                    |
| ---------------- | ------------------------------------ | ---------------------------------------- |
| Civility         | Is it respectful?                    | 0 = attacks a person in the conversation |
| Relevance        | Is it on topic?                      |                                          |
| Reasoning        | Does it support its claims?          | Can be N/A when the reply makes no claim |
| Constructiveness | Does it move the discussion forward? |                                          |

The scorer sees the comment that started the thread and the comment being
replied to. Six edge-case rules cover sarcasm vs. personal attacks, swearing,
provocation, dismissal, moderator outcomes, and insults at people outside the
conversation. Rules 1–5 came from scoring a 10-reply pilot. Rule 6 came from
the 60-reply review.

**Conversation-level quality** is the average of a conversation's reply scores
on each dimension (ignoring N/A). It is not labeled separately.

## 2. Reference labels

- **60 replies** from the CGA-CMV training split: 20 replies moderators
  removed, 10 earlier replies from conversations that later derailed, and 30
  replies from conversations that stayed civil (random seed 2026; the 10 pilot
  replies excluded) were human reviewed.

### Sanity check against moderator removals

If the rubric measures civility, replies moderators removed should score lower:

| Group                                                | Replies | Average civility (reference) | Average civility (Opus) |
| ---------------------------------------------------- | ------- | ---------------------------- | ----------------------- |
| Conversations that stayed civil                      | 30      | 1.87                         | 1.83                    |
| Earlier replies in conversations that later derailed | 10      | 1.50                         | 1.60                    |
| Replies moderators removed                           | 20      | 1.15                         | 1.15                    |

The order matches. But 6 of the 20 removed replies scored civility 2:
moderator removals are a noisy signal, so the dataset's civil/derailed label
is not a substitute for the rubric.

## 3. LLM labeler

Code: [`artifacts/quality_labeler.py`](../artifacts/quality_labeler.py) and
[`scripts/label_quality.py`](../scripts/label_quality.py) (tests:
`tests/test_quality_labeler.py`).

- The system prompt is a short instruction plus the full rubric file.
- Output is constrained to a JSON schema: a one-sentence rationale and the
  four scores. Refusals, errors and invalid answers are recorded as unscored,
  never guessed.
- Two ways to run it: through the Claude Code CLI (`--backend claude-code`,
  uses a logged-in Claude account, no API key) or the Anthropic API
  (`--backend api`, plus the half-price Batches API for large runs).
- Settings for every result below: effort `low`, Claude Code backend.

Reproduce:

```
python scripts/label_quality.py evaluate --model claude-opus-5-5
python scripts/label_quality.py evaluate --model claude-haiku-5-5
```

## 4. Agreement with the reference labels

**Exact** = share of the 60 replies with the same score. **Kappa** = Cohen's
kappa, agreement corrected for chance (1 = perfect, 0 = chance). Both models
were within one point of the reference on 97–100% of replies for every
dimension.

| Dimension                 | Haiku 5.5 exact / kappa | Opus 5.5 exact / kappa (run 1) | Opus 5.5 exact / kappa (run 2) |
| ------------------------- | ----------------------- | ------------------------------ | ------------------------------ |
| Civility                  | 0.93 / 0.87             | 0.93 / 0.87                    | 0.93 / 0.87                    |
| Relevance                 | 0.98 / 0.85             | 0.92 / 0.51                    | 0.93 / 0.57                    |
| Reasoning                 | 0.58 / 0.41             | 0.73 / 0.61                    | 0.73 / 0.60                    |
| Constructiveness          | 0.78 / 0.63             | 0.85 / 0.74                    | 0.87 / 0.77                    |
| List price for 60 replies | $0.03                   | $0.94                          | $0.93                          |

**Opus agrees with itself** across the two runs on 97% (civility), 98%
(relevance), 93% (reasoning) and 92% (constructiveness) of replies, so the
gaps above are mostly real disagreement with the reference, not randomness.

**Choice:** Opus 5.5 for the labeling run. It is clearly better on the two
hard dimensions. Haiku is about 30× cheaper and equal on civility, so it is a
reasonable choice if only civility is needed.

**Relevance kappa looks low but isn't alarming.** 57 of 60 reference labels
are 2, and kappa penalizes disagreement heavily when one label dominates. The
exact agreement (92–98%) is the more meaningful number. It also means
relevance barely separates comments on CGA-CMV.

## 5. Where the LLM disagrees (error review)

Disagreements for each run are saved as `disagreements.csv` in each
`quality-labels/eval-*` folder, with the reference note and the LLM's
rationale side by side.

**Civility (4 of 60, Opus).**

- #7 and #30 were flagged as borderline when the reference labels were made
  (implying someone is a bigot; "salty bitch" aimed at a generic "you"). Opus
  scored each one point milder.
- #42, a polite exit from the discussion, was scored 1 by Opus for being
  "somewhat dismissive".
- **#37 is a reference-label error.** Calling the Koch brothers "Super Evil"
  insults people outside the conversation, which rule 6 caps at civility 1.
  Opus applied the rule; the reference label (2) did not.

**Reasoning (16 of 60, Opus run 1).** Two patterns:

- **Questions scored 0 instead of N/A (5 cases).** The LLM treats pointed
  questions as implied claims with no support. Haiku did this even more,
  which explains most of its low reasoning score.
- **1 vs. 2 on thin support (8 cases), in both directions.** Where "some
  support" ends and "clear reasons" begin is the most subjective line in the
  rubric.

**Relevance (5 of 60, Opus).** Always one point lower than the reference:
Opus marks side arguments and tangents as 1 where the reference gave 2.

**Constructiveness (9 of 60, Opus).** No consistent direction (5 higher, 4
lower).

## 6. Labeling run

**Status (2026-11-21): partial.** The run uses the Claude Code plan's usage
limit, which stopped it twice. So far **1,582 of 5,134 replies** (311
conversations) are labeled with no outstanding failures. It's resumable, so
re-running the command below continues it. Numbers below are from the partial
set and will change slightly when the run finishes.

| Dimension | 0 | 1 | 2 | N/A |
|---|---|---|---|---|
| Civility | 4% | 37% | 59% | |
| Relevance | 0% | 6% | 94% | |
| Reasoning | 18% | 49% | 28% | 5% |
| Constructiveness | 8% | 36% | 56% | |

**Sanity check at scale:** replies in derailed conversations average civility
1.47 (n = 851) vs. 1.65 in civil ones (n = 731), and score 0 more than twice
as often (5.4% vs. 2.2%). That's the expected direction, and matches the
reference-set check in section 2.

- **Sample:** 1,000 whole conversations (5,134 replies), drawn as matched
  civil/derailed pairs from all three splits, excluding the 60 reference
  replies. Whole conversations are needed for conversation-level averages.
- **Command:** `python scripts/label_quality.py label --conversations 1000 --model claude-opus-5-5`
- **Output:** `data/local/quality_labels/llm_labels-claude-opus-5-5.csv`
  (gitignored; holds IDs, split, derailed flag, scores and rationale).

## 7. Lightweight scorer (first version)

The LLM is too slow and usage-hungry to score every comment the pipeline sees,
so a cheap model is trained on the LLM labels and called from the pipeline.

- **Code:** `artifacts/quality_scorer.py` (`QualityScorer`, `score_comments`),
  trained by `scripts/train_quality_scorer.py`, tested in
  `tests/test_quality_scorer.py`.
- **Model:** per dimension, ridge regression on TF-IDF word 1–2 grams of the
  reply plus five simple features (length, question marks, second-person
  words, TF-IDF similarity to parent and to thread start). Outputs a 0–2 float
  per dimension, `reasoning_na_prob` for "no claim", and `overall_score` (mean
  of civility, reasoning, constructiveness; relevance left out, see below).
- **Training data:** the 1,251 Opus labels available on 2026-11-21 (233
  conversations). Ridge strength chosen by 5-fold cross-validation on the LLM
  labels, grouped by CGA-CMV pair. The 60 reference replies were **not** used
  for training or tuning.
- **Saved model:** `data/local/quality_scorer.joblib` (gitignored; retrain
  with the script).

**Results on the 60 reference replies** (MAE = mean absolute error on the 0–2
scale, lower is better; "predict mean" = MAE of always guessing the average,
for scale; reasoning excludes the 11 N/A references):

| Dimension | Scorer MAE | Scorer Spearman | Predict-mean MAE | Opus MAE | Opus Spearman |
|---|---|---|---|---|---|
| Civility | 0.47 | 0.33 | 0.53 | 0.07 | 0.90 |
| Relevance | 0.10 | 0.30 | 0.10 | 0.08 | 0.58 |
| Reasoning | 0.46 | 0.50 | 0.51 | 0.20 | 0.72 |
| Constructiveness | 0.48 | 0.47 | 0.60 | 0.15 | 0.81 |

Reasoning N/A was called correctly for 80% of replies (Opus: 90%).
Cross-validated against the LLM labels the numbers are similar (civility MAE
0.46 / Spearman 0.36; reasoning 0.40 / 0.69; constructiveness 0.42 / 0.57), so
the scorer is not overfitting the reference set. It just hasn't learned much yet.

**Error analysis** (`reports/quality-labels/scorer/scorer_errors.csv`; 22 of
60 replies are off by at least 0.75 on some dimension):

- **Pulled toward the middle.** Scores cluster around 1.2–1.9. Clear personal
  attacks (reference civility 0) get about 1.2–1.3, and some plainly civil
  replies get about 1.2. Ridge regression on about 1,250 examples, most of them
  civil, does this.
- **Civility needs meaning, not words.** The missed attacks are implied ("implies
  the other user is a bigot"), not profanity. Word counts can't see them.
- **Short concessions and questions** ("thanks, fair point", probing questions)
  are under-scored on constructiveness. They're short, and short replies are
  usually low-effort in the training data.

**Tried and not kept:** sentence embeddings (all-MiniLM-L6-v2,
all-mpnet-base-v2) in place of TF-IDF. Cross-validated MAE was within about
0.03 of TF-IDF on every dimension, so they don't justify a 100–400 MB model
download in the pipeline.

**Relevance should be dropped or merged.** 94% of LLM labels and 57 of 60
reference labels are 2. Neither the LLM (kappa 0.51) nor the scorer (no better
than predicting the mean) separates the rare 1s reliably. That's too little
variation to be a useful score on CGA-CMV. The scorer still outputs it, but
`overall_score` leaves it out.

**Next:** retrain once the labeling run finishes (about 4× the data). If
civility is still weak, combine with the baseline civility model from Task 6
(trained on moderator removals) or fine-tune a small transformer.

## 8. Limitations and next steps

- **The 60 aren't a perfectly clean held-out set.** They were used to compare
  rubric v1 against v2 (section 4), and Claude Opus drafted them before human
  review. A freshly hand-labeled batch would give a cleaner agreement number.
- **Data-sharing permission still to confirm.** The issue asks to confirm we
  may send Reddit text to an external LLM; this has not been recorded yet.
- **60 replies is small.** One reply is about 1.7 percentage points of exact
  agreement.
- **Only CGA-CMV.** All replies are from r/changemyview arguments. Other
  sources (Coarse Discourse, GitHub, Discord) may need rubric changes,
  especially relevance.
- **Effort was `low`.** Higher effort may improve reasoning agreement at a
  higher cost. That's untested.
- **For Task #6:** the LLM labels can train a cheaper model to predict rubric
  scores, or be averaged into conversation-level quality features.
