import re
import json
from typing import Dict
import os

from tqdm import tqdm
from dotenv import load_dotenv
import matplotlib.pyplot as plt

from client.models import Query, QueryResponse
from client.query import query_model

INVALID_ANS = "[invalid]"

def standard_prompt_template(question: str) -> str:
    """
    Converts a gsm8k question into a standard model input

    Args:
        question: gsm8k question.
    Returns:
        prompt for a model to answer input question.
    """

    prompt = f"""Output a numerical answer to the following problem with two or fewer steps of reasoning. Output your numerical
answer as the only line of your output in the format "#### <numerical_answer>."

Problem: {question}
""".strip()

    return prompt

def standard_output_extractor(model_generation: str) -> str:
    """
    Extracts the string answer from a model generation, assuming it was prompted 
    using a prompt from `standard_prompt_template`.

    Args:
        model_generation: the string generation from the model
    Returns:
        String representing the numerical output of the model for the question, or "[invalid]" if
            no output can be extracted.
    """

    ANS_RE = re.compile(r"#### (\-?[0-9\.\,]+)")

    match = ANS_RE.search(model_generation)

    if match:
        match_str: str = match.group(1).strip()
        match_str = match_str.replace(",", "")
        return match_str
    else:
        return INVALID_ANS


# ------------------------------------------- #
# TODO For you to fill in 
# ------------------------------------------- #



def _evaluate(models, prompt_template, output_path):
    """Persist each completed request immediately; never overwrite prior evidence."""
    from pathlib import Path
    from decimal import Decimal, InvalidOperation

    data_path = Path(__file__).parent / "data" / "gsm8k_first_100.jsonl"
    examples = [json.loads(line) for line in data_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(examples) != 100:
        raise ValueError("Expected the official 100-example GSM8K subset")
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.with_suffix(".summary.json").exists():
        raise FileExistsError(output.with_suffix(".summary.json"))
    totals = {model: {"correct": 0, "invalid": 0, "count": 0, "cost": 0.0} for model in models}
    # Exclusive creation prevents accidental duplicate billing on a blind rerun.
    with output.open("x", encoding="utf-8") as handle:
        for model in models:
            for index, example in enumerate(tqdm(examples, desc=model)):
                prompt = prompt_template(example["question"])
                response = query_model(model_id=model, query=Query(turns=[{"user": prompt}]))
                extracted = standard_output_extractor(response.text)
                valid = False
                correct = False
                try:
                    predicted = Decimal(extracted)
                    expected = Decimal(str(example["numerical_answer"]).replace(",", ""))
                    valid = predicted.is_finite() and expected.is_finite()
                    correct = valid and predicted == expected
                except InvalidOperation:
                    pass
                record = {"model": model, "index": index, "question": example["question"],
                          "prompt": prompt, "response": response.model_dump(),
                          "expected": example["numerical_answer"], "extracted": extracted,
                          "valid": valid, "correct": correct}
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                handle.flush()
                stats = totals[model]
                stats["count"] += 1
                stats["correct"] += int(correct)
                stats["invalid"] += int(not valid)
                stats["cost"] += response.cost
    for stats in totals.values():
        stats["accuracy"] = stats["correct"] / stats["count"]
    summary = output.with_suffix(".summary.json")
    with summary.open("x", encoding="utf-8") as handle:
        json.dump(totals, handle, indent=2)
    return totals


def eval_model_on_gsm8k() -> None:
    """
    Benchmark models A and B on the GSM8K dataset using the standard prompt template.
    
    See example_usage.py for how to query models and handle responses.
    The data file (gsm8k_first_100.jsonl) contains 'question' and 'numerical_answer' fields.
    
    Think about: What metric will you use to evaluate performance? How will you 
    handle cases where the model's output cannot be parsed?
    """
    return _evaluate(("A", "B"), standard_prompt_template, "results/gsm8k-standard.jsonl")



def superior_prompt_template(question: str) -> str:
    """
    Design your own prompt template that outperforms standard_prompt_template on model A.
    
    Args:
        question: gsm8k question.
    Returns:
        Your improved prompt for the model.
    
    Look at standard_prompt_template() to understand the baseline approach. What 
    aspects of how you prompt the model might affect its reasoning or accuracy?
    
    NOTE: Your prompt must still produce output in the "#### <answer>" format
    so that standard_output_extractor() can parse the response.
    """
    return (
        "Solve the arithmetic problem carefully. Identify the quantities and units, "
        "compute intermediate values, then check the result against the question. "
        "Do not round intermediate calculations. Give a short explanation followed "
        "by a final line containing only #### <numerical_answer>, with no units "
        "or trailing punctuation.\n\nProblem: " + question
    )

def eval_model_on_gsm8k_with_improved_prompt() -> None:
    """
    Evaluate model A using your superior_prompt_template.
    """
    return _evaluate(("A",), superior_prompt_template, "results/gsm8k-improved.jsonl")

def plot_comparison(standard_path="results/gsm8k-standard.jsonl",
                    improved_path="results/gsm8k-improved.jsonl",
                    output_path="results/gsm8k-comparison.png"):
    """Compare paired, complete raw evaluations; never infer improvement from a prompt."""
    from pathlib import Path
    def read(path, models):
        rows = [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines()]
        keyed = {(r["model"], r["index"]): r for r in rows}
        if len(keyed) != len(rows) or set(keyed) != {(m, i) for m in models for i in range(100)}:
            raise ValueError("Require exactly 100 unique examples per expected model")
        if any(type(r["correct"]) is not bool for r in rows):
            raise ValueError("Correctness must be a JSON boolean")
        return keyed
    baseline = read(standard_path, ("A", "B"))
    improved = read(improved_path, ("A",))
    for index in range(100):
        records = [baseline["A", index], baseline["B", index], improved["A", index]]
        if len({(r["question"], str(r["expected"])) for r in records}) != 1:
            raise ValueError("Cannot compare mismatched examples or expected answers")
    scores = [sum(baseline[m, i]["correct"] for i in range(100)) / 100 for m in ("A", "B")]
    scores.append(sum(improved["A", i]["correct"] for i in range(100)) / 100)
    destination = Path(output_path)
    if destination.exists():
        raise FileExistsError(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots()
    bars = ax.bar(["A standard", "B standard", "A improved prompt"], scores)
    ax.set(ylim=(0, 1), ylabel="Exact numerical accuracy (all 100 questions)")
    ax.bar_label(bars, labels=[f"{score:.0%}" for score in scores], padding=3)
    fig.tight_layout()
    fig.savefig(destination)
    plt.close(fig)
    return {"A_standard": scores[0], "B_standard": scores[1], "A_improved": scores[2],
            "A_paired_accuracy_difference": scores[2] - scores[0]}


if __name__=="__main__":
    import argparse
    parser = argparse.ArgumentParser(description="GSM8K evaluation; standard/improved make paid Vertex calls")
    parser.add_argument("mode", choices=("standard", "improved", "plot"))
    parser.add_argument("--allow-paid-calls", action="store_true")
    args = parser.parse_args()
    load_dotenv()
    if args.mode != "plot" and not args.allow_paid_calls:
        parser.error("Real API calls require explicit --allow-paid-calls and configured GCP billing/ADC")
    if args.mode == "standard":
        eval_model_on_gsm8k()
    elif args.mode == "improved":
        eval_model_on_gsm8k_with_improved_prompt()
    else:
        print(json.dumps(plot_comparison(), indent=2))
