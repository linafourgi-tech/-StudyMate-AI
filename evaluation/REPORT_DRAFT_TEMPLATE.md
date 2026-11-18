# Suggested Role 7 report subsection (replace [brackets] with measured data)

## Evaluation methodology
We prepared [N] test cases from [specific permitted lecture files]. The test set covered [topics], including [N] missing-information and difficult cases. For reproducibility, the benchmark records the method, prompt, model/configuration, number of cases, response time, output validity and whether the provided lecture supports each answer. A deterministic template method using manually annotated facts was used as a simple baseline. [Describe full-context/retrieval comparisons actually performed and how they used the same source material.]

We used an 8-point question-quality rubric covering grounding, correctness, clarity, and distractor quality, each scored 0–2. Feedback was evaluated on a 6-point rubric covering grading, evidence, and usefulness. [N] outputs were independently reviewed by [N] reviewers. The automated format validator checked only output structure; factual quality required human review.

## Results
- Test cases evaluated: [N].
- Question-quality mean: [N/A or X/8] for [method and N rated cases].
- Feedback-quality mean: [N/A or X/6].
- MCQ format validity: [N/A or %] across [N] cases.
- Missing-information refusal accuracy: [N/A or %] across [N] cases.
- Mean end-to-end quiz latency: [N/A or ms] for [method and N attempts].
- Provider/model and token costs: [verified price source/date and result] or N/A.
- Manual quiz-preparation time: [minutes and process].
- StudyMate preparation + checking time: [minutes and process].
- Time saved: [formula output or N/A].

## Limitations and observed failure
We observed [specific case ID, actual failure, root-cause evidence]. The likely cause was [verified evidence, not speculation]. [State workaround or proposed mitigation]. The synthetic fixture results are unit-test demonstrations, not field-study outcomes. Any learner-simulation results from Role 5 are labelled as simulated.

## Verification
Tests executed with `python -m pytest tests/test_benchmarks.py -v`: [actual output and date]. Model experiment command: [actual command]. Input lecture versions / file hashes: [actual values]. Do not claim that unittest pass counts are human-rated question quality.
