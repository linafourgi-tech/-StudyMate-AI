# Role 7: human quality rubric (not an automatic truth detector)

**Reviewer instructions:** Check the generated question against its lecture text, then choose 0, 1, or 2 for each criterion. Record a short justification and reviewer ID. Do not mark unanswered categories as zero: leave them blank / `null` until reviewed.

| Question criterion | 0 | 1 | 2 |
|---|---|---|---|
| Grounding | Not supported by lecture | Partially supported / vague | Fully supported by the supplied lecture |
| Correctness | Wrong answer or misinformation | Partly correct / ambiguous | Correct, unique answer |
| Clarity | Confusing or impossible to understand | Understandable with minor issues | Clear and specific question |
| Distractors | Missing, duplicates, or clearly invalid | Some distractors are too easy | Three distinct, plausible wrong answers |

**Question quality:** `(grounding + correctness + clarity + distractors) / 8 × 100%`.

| Feedback criterion | 0 | 1 | 2 |
|---|---|---|---|
| Grading | Incorrect student score | Partly correct | Correct score relative to reference |
| Evidence | Unsupported or misleading explanation | Partial citation/evidence | Explanation demonstrably supported by lecture |
| Usefulness | No actionable explanation | Vague advice | Explains why, teaches correct concept |

**Feedback quality:** `(grading + evidence + usefulness) / 6 × 100%`.

Optional: use two independent reviewers for a subset and report agreement; investigate disagreements. Rubric scores describe a human judgment, not automated ground truth. Scoring can be subjective.
