"""
Prompt builder for ML-assisted abstract screening.

Implements the Abstract ScreenPrompt pattern from:
  Cao et al. (2025). "Development of Prompt Templates for Large Language
  Model-Driven Screening in Systematic Reviews." Ann Intern Med.
  doi:10.7326/ANNALS-24-02189

The prompt uses Framework Chain-of-Thought (CoT) reasoning: the model is given
study objectives, numbered inclusion/exclusion criteria, and instructed to
evaluate each criterion step-by-step before making a YYY (include) or
XXX (exclude) decision.
"""

from textwrap import dedent

FEWSHOT = """
Example (FORMAT TO COPY EXACTLY):

Reasoning:
The abstract does not include patients/public; it focuses on algorithm performance in radiology. No attitudes/trust. Not an education/training context.

Decision:
XXX
""".strip()


def build_abstract_prompt(
    *,
    title: str,
    abstract: str,
    review_objectives: str,
    inclusion: list[str],
    exclusion: list[str],
) -> str:
    """Build a screening prompt for a single title/abstract pair.

    Args:
        title: Paper title.
        abstract: Paper abstract text.
        review_objectives: Free-text description of the review's objectives.
        inclusion: List of inclusion criteria strings.
        exclusion: List of exclusion criteria strings.

    Returns:
        A formatted prompt string ready to send to the model.
    """
    inc_block = "\n".join(f"{i+1}. {line}" for i, line in enumerate(inclusion))
    exc_block = "\n".join(f"{i+1}. {line}" for i, line in enumerate(exclusion))

    return dedent(f"""
    You are screening titles and abstracts for a scoping review.

    Study Objectives:
    {review_objectives.strip()}

    Inclusion Criteria:
    {inc_block}

    Exclusion Criteria:
    {exc_block}

    Paper:
    Title: {title.strip()}
    Abstract: {abstract.strip()}

    Reasoning Plan (≤5 sentences):
    1) Restate the main topic in one sentence.
    2) Evaluate each inclusion criterion (met / unclear / not met).
    3) Evaluate each exclusion criterion (applies / unclear / does not apply).
    4) Decide per rules:
       - If ANY inclusion is met and no exclusion clearly applies → YYY.
       - If information is insufficient/ambiguous → YYY (Include/Uncertain).
       - If ≥1 exclusion clearly applies and inclusions fail → XXX.

    Output FORMAT (STRICT):
    - First write a short 'Reasoning:' paragraph (≤5 sentences).
    - Then write 'Decision:' on its own line.
    - On the VERY LAST LINE of the entire output, write ONLY one token: YYY or XXX.
    - Do NOT add punctuation or any other text after the token.

    {FEWSHOT}
    """).strip()
