# ML-Assisted Abstract Screening Pipeline for Scoping and Systematic Reviews

A reusable, open-source machine learning (ML) pipeline that screens titles and abstracts for scoping and systematic reviews. Designed so that **no coding knowledge is needed** - you configure your review criteria in a single file and run one command.

This pipeline implements the *Abstract ScreenPrompt* methodology from Cao et al. (2025), which uses Framework Chain-of-Thought (CoT) prompting to achieve high sensitivity (97.7%) and specificity (85.2%) across diverse systematic reviews.

## Citation

If you use this pipeline in your research, please cite:

> DOI Pending

The prompting methodology is based on:

> Cao C, Sang J, Arora R, et al. Development of Prompt Templates for Large Language Model-Driven Screening in Systematic Reviews. *Ann Intern Med*. 2025;178(3):389-401. doi:10.7326/ANNALS-24-02189

## How It Works

1. You export your search results from Covidence (or any reference manager) as a CSV file containing titles and abstracts.
2. You fill in `config.yaml` with your review's objectives, inclusion criteria, and exclusion criteria.
3. The pipeline sends each abstract to an OpenAI model with a structured prompt that instructs the model to evaluate each criterion step-by-step.
4. The model outputs a reasoned decision: **YYY** (include/uncertain) or **XXX** (exclude).
5. Results are saved to a CSV file you can import back into Covidence or use directly.

The pipeline defaults to **including uncertain records** (YYY) to minimize false exclusions, consistent with best practices for systematic review screening where sensitivity is prioritized over specificity.

## Prerequisites

- **Python 3.8 or higher** - download from [python.org](https://www.python.org/downloads/) if you don't have it
- **An OpenAI API key** - sign up at [platform.openai.com](https://platform.openai.com/) and create an API key
- **Your abstracts in CSV format** - exported from Covidence, PubMed, or any reference manager (must have `Title` and `Abstract` columns)

## Setup

### Step 1: Download this repository

Click the green **Code** button above, then **Download ZIP**. Unzip the folder, or clone it:

```bash
git clone https://github.com/shawnasadian84-dot/ml-abstract-screening.git
cd ml-abstract-screening
```

### Step 2: Install Python dependencies

Open a terminal (Mac: Terminal app; Windows: Command Prompt or PowerShell), navigate to the folder, and run:

```bash
pip install -r requirements.txt
```

### Step 3: Add your OpenAI API key

Copy the example environment file and add your key:

```bash
cp .env.example .env
```

Open `.env` in any text editor and replace `your-api-key-here` with your actual API key:

```
OPENAI_API_KEY=sk-proj-abc123...
```

> **Important:** Never share your `.env` file or commit it to GitHub. The `.gitignore` file already prevents this.

## Configuration

Open `config.yaml` in any text editor. Fill in three sections:

### 1. Review Objectives

Describe what your review is about. Be specific - the model uses this to understand the scope of your review.

```yaml
review_objectives: >
  To systematically review the effectiveness of telehealth interventions
  on patient outcomes in chronic disease management.
```

### 2. Inclusion Criteria

List each criterion on its own line, starting with a dash and space (`- `). Follow the PCC (Population, Concept, Context) or PICO framework for clarity.

```yaml
inclusion_criteria:
  - "Population: Adults (>=18) with chronic diseases."
  - "Concept: Telehealth or telemedicine interventions."
  - "Context: Primary care or outpatient settings."
  - "Study design: RCTs, cohort studies, or systematic reviews."
  - "Language: English."
  - "Date range: January 2010 - December 2025."
```

### 3. Exclusion Criteria

```yaml
exclusion_criteria:
  - "Pediatric populations only (<18 years)."
  - "Purely technical papers without patient outcomes."
  - "Conference abstracts without sufficient methodological detail."
  - "Non-English publications."
```

### Model Settings

The defaults work well for most use cases. You can adjust these optionally:

| Setting | Default | Description |
|---------|---------|-------------|
| `model` | `gpt-4o` | OpenAI model. `gpt-4o-mini` is cheaper but slightly less accurate. |
| `temperature` | `0` | Set to 0 for deterministic, reproducible results. |
| `max_output_tokens` | `300` | Max response length per abstract. 300 is sufficient. |

## Running the Pipeline

### Basic usage

```bash
python screen.py --csv your_abstracts.csv
```

This screens **all** abstracts in the CSV and saves results to `decisions.csv`.

### Custom output file

```bash
python screen.py --csv your_abstracts.csv --out_csv my_results.csv
```

### Override the model

```bash
python screen.py --csv your_abstracts.csv --model gpt-4o-mini
```

## Batch Processing

For large datasets (hundreds or thousands of abstracts), you can process records in batches using the `--limit` flag. This is useful for:

- **Cost control** - review a small batch of results before committing to the full run
- **Monitoring quality** - check that the model is making reasonable decisions on your criteria
- **Crash resilience** - progress is saved after every single record, so nothing is lost

### How it works

```bash
# First run: screens the first 25 abstracts
python screen.py --csv your_abstracts.csv --limit 25

# Second run (same command): detects 25 are done, screens the next 25
python screen.py --csv your_abstracts.csv --limit 25

# Repeat until all abstracts are screened
```

The pipeline automatically detects which records have already been processed in the output file and skips them. You just re-run the same command and it picks up where it left off.

### Recommended workflow

1. **Test first:** Run with `--limit 5` and review the output to make sure the decisions and reasoning look reasonable for your criteria.
2. **Adjust if needed:** If results seem off, refine your criteria in `config.yaml` and delete the output CSV to start fresh.
3. **Scale up:** Once satisfied, run with `--limit 25` or `--limit 50` in batches, or remove `--limit` entirely to process everything.

## Output

The pipeline produces a CSV file (default: `decisions.csv`) with three columns:

| Column | Description |
|--------|-------------|
| `record_id` | The record identifier from your CSV (Covidence #, Accession Number, or row number) |
| `decision` | `YYY` (include for full-text review) or `XXX` (exclude) |
| `rationale` | The model's step-by-step reasoning for the decision |

### Interpreting results

- **YYY (Include):** The abstract met at least one inclusion criterion and no exclusion criterion clearly applied. These records should advance to full-text review.
- **XXX (Exclude):** At least one exclusion criterion clearly applied and inclusion criteria were not met.
- **When ambiguous:** The pipeline defaults to YYY (include) to avoid missing relevant studies. This prioritizes **sensitivity** over **specificity**, consistent with systematic review best practices.

> **Important:** This pipeline is intended to assist, not replace, human screening. All decisions should be verified by human reviewers, particularly records near the decision boundary. The `rationale` column helps you quickly identify records that may need closer manual review.

## Input CSV Format

Your CSV needs at minimum two columns: `Title` and `Abstract` (case-insensitive). The pipeline also recognizes these optional ID columns:

- `Covidence #` (if exported from Covidence)
- `Accession Number`
- `record_id`

If none of these are present, the pipeline uses the row number as the record ID.

### Exporting from Covidence

1. In your Covidence review, go to **Screen** > **Title and Abstract screening**
2. Click **Export** > **Export as CSV**
3. Save the file and use it as the `--csv` input

## Tips

- **Start small.** Always test with `--limit 5` first to verify the model is interpreting your criteria correctly.
- **Be specific in your criteria.** Vague criteria lead to inconsistent screening. Use the PCC or PICO framework.
- **Review the rationale column.** It tells you *why* each decision was made, which helps you spot systematic issues.
- **Cost awareness.** Each abstract costs approximately $0.001-$0.01 depending on the model. A full run of 1,000 abstracts with `gpt-4o` costs roughly $2-$5.
- **Reproducibility.** With `temperature: 0`, results are deterministic. Running the same abstracts again produces the same decisions.

## File Overview

```
├── config.yaml            ← Edit this: your review criteria and settings
├── screen.py              ← Main script: run this to screen abstracts
├── prompt_templates.py    ← Prompt builder (no need to edit)
├── .env.example           ← Copy to .env and add your API key
├── requirements.txt       ← Python dependencies
├── .gitignore             ← Prevents committing sensitive files
├── LICENSE                ← MIT License
└── README.md              ← This file
```

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
