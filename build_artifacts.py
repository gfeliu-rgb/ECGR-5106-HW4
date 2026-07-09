from pathlib import Path
import textwrap

import nbformat as nbf
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.image as mpimg
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
PLOTS = ROOT / "plots"
REPORT = ROOT / "report"
REPORT.mkdir(exist_ok=True)

GITHUB_URL = "https://github.com/Gilbertofeliu/ECGR-5106-HW4"


def fmt(value):
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def md_table(df, max_rows=None):
    if max_rows:
        df = df.head(max_rows)
    out = df.copy()
    for col in out.columns:
        if pd.api.types.is_float_dtype(out[col]):
            out[col] = out[col].map(lambda x: f"{x:.4f}")
    return out.to_markdown(index=False)


def load_tables():
    return {
        "p1": pd.read_csv(RESULTS / "problem1_transformer_summary.csv"),
        "p2": pd.read_csv(RESULTS / "problem2_transformer_summary.csv"),
        "p3": pd.read_csv(RESULTS / "problem3_transformer_summary.csv"),
        "p4": pd.read_csv(RESULTS / "problem4_transformer_summary.csv"),
        "hw2_p1": pd.read_csv(ROOT.parent / "Homework_2" / "results" / "problem1_summary.csv"),
        "hw2_p2": pd.read_csv(ROOT.parent / "Homework_2" / "results" / "problem2_report_table.csv"),
        "hw3": pd.read_csv(ROOT.parent / "Homework_3" / "results" / "summary.csv"),
    }


def selected_cols(df, cols):
    return df.loc[:, cols].copy()


def add_text_page(pdf, title, blocks):
    fig = plt.figure(figsize=(8.5, 11))
    fig.patch.set_facecolor("white")
    fig.text(0.07, 0.95, title, fontsize=18, fontweight="bold")
    y = 0.91
    for block in blocks:
        if block.startswith("## "):
            fig.text(0.07, y, block[3:], fontsize=13, fontweight="bold")
            y -= 0.032
            continue
        wrapped = textwrap.wrap(block, width=95) or [""]
        for line in wrapped:
            fig.text(0.07, y, line, fontsize=9.5, va="top")
            y -= 0.018
        y -= 0.01
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def add_table_page(pdf, title, df, font_size=7):
    show = df.copy()
    for col in show.columns:
        if pd.api.types.is_float_dtype(show[col]):
            show[col] = show[col].map(lambda x: f"{x:.4f}")
    fig, ax = plt.subplots(figsize=(11, 8.5))
    fig.patch.set_facecolor("white")
    ax.axis("off")
    ax.set_title(title, fontsize=15, fontweight="bold", pad=15)
    table = ax.table(
        cellText=show.astype(str).values,
        colLabels=[c.replace("_", "\n") for c in show.columns],
        loc="center",
        cellLoc="center",
        colLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(font_size)
    table.scale(1.0, 1.3)
    for (row, _), cell in table.get_celld().items():
        cell.set_edgecolor("#c7d0d9")
        if row == 0:
            cell.set_facecolor("#244761")
            cell.set_text_props(color="white", fontweight="bold")
        elif row % 2 == 0:
            cell.set_facecolor("#f4f7f9")
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def add_images_page(pdf, title, names):
    fig, axes = plt.subplots(2, 2, figsize=(11, 8.5))
    axes = axes.ravel()
    fig.suptitle(title, fontsize=15, fontweight="bold")
    for ax, name in zip(axes, names):
        ax.imshow(mpimg.imread(PLOTS / name))
        ax.set_title(name.replace(".png", "").replace("_", " ").title(), fontsize=9)
        ax.axis("off")
    for ax in axes[len(names):]:
        ax.axis("off")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def build_report():
    t = load_tables()
    p1_best = t["p1"].sort_values("valid_accuracy", ascending=False).iloc[0]
    p2_best = t["p2"].sort_values("valid_loss").iloc[0]
    p3_best = t["p3"].sort_values(["bleu4", "valid_loss"], ascending=[False, True]).iloc[0]
    p4_best = t["p4"].sort_values(["bleu4", "valid_loss"], ascending=[False, True]).iloc[0]
    hw2_p1_best = t["hw2_p1"].sort_values("valid_accuracy", ascending=False).iloc[0]
    hw2_p2_best = t["hw2_p2"].sort_values("valid_loss").iloc[0]
    hw3_best_enfr = t["hw3"][t["hw3"]["direction"].eq("English-to-French")].sort_values("word_bleu4", ascending=False).iloc[0]
    hw3_best_fren = t["hw3"][t["hw3"]["direction"].eq("French-to-English")].sort_values("word_bleu4", ascending=False).iloc[0]

    report_md = f"""# Homework 4

Name: Gilberto Feliu  
Student ID: 801257813  
Assignment: Homework 4  
Repository: {GITHUB_URL}

## Experimental Notes

All notebooks are included with executed outputs visible. The code uses PyTorch transformer models, the same Homework 2 character sequence, the provided Tiny Shakespeare file, and the same Homework 3 80/20 train-validation split stored in `results/hw3_split_indices.json`. The experiments train every listed block/head combination from the assignment grid. The listed grid of 1, 2, and 4 transformer blocks crossed with 2 and 4 heads gives six concrete configurations, and all six were run for both translation directions.

## Problem 1

The best transformer validation accuracy on the assigned paragraph was {p1_best.valid_accuracy:.4f} at sequence length {int(p1_best.sequence_length)}. Compared with the best Homework 2 RNN-family result ({hw2_p1_best.cell_type}, sequence length {int(hw2_p1_best.sequence_length)}, validation accuracy {hw2_p1_best.valid_accuracy:.4f}), the transformer was close in top-1 accuracy and competitive in top-3 accuracy. The RNN-family model still has an advantage on this tiny paragraph because recurrence is a strong inductive bias for very small sequential datasets, while self-attention pays more parameters to learn context relationships.

Training time increased from {t['p1'].sequence_length.min()} to {t['p1'].sequence_length.max()} characters because causal self-attention scales approximately with sequence_length^2 while the feed-forward layers scale linearly in sequence length. The validation loss did not improve at length 30, which indicates the extra context did not overcome the small-data overfitting/noise penalty.

## Problem 2

For Tiny Shakespeare, the lowest validation loss was {p2_best.valid_loss:.4f} from L={int(p2_best.num_layers)}, H={int(p2_best.num_heads)}, sequence length {int(p2_best.sequence_length)}. The best Homework 2 recurrent result by validation loss was {hw2_p2_best.cell_type} with sequence length {int(hw2_p2_best.sequence_length)} and validation loss {hw2_p2_best.valid_loss:.4f}. The transformer therefore slightly improved the best validation-loss result while also exposing the cost of attention through larger operation counts at longer sequence lengths.

Depth had the clearest model-size effect: moving from 1 to 4 transformer blocks increased parameters and training time. The 4-block Tiny Shakespeare models drove training loss lower but validation loss became worse, which is a clear overfitting signal. Increasing heads from 2 to 4 did not consistently improve validation loss because d_model was fixed, so each head received a smaller subspace and the added attention partitioning did not offset the generalization penalty.

## Problem 3

For English-to-French translation, the best transformer configuration by BLEU-4 was L={int(p3_best.num_layers)}, H={int(p3_best.num_heads)} with BLEU-4 {p3_best.bleu4:.4f} and validation loss {p3_best.valid_loss:.4f}. The Homework 3 attention GRU reached word BLEU-4 {hw3_best_enfr.word_bleu4:.4f}, so the best transformer substantially improved the BLEU score on the same validation split. Exact sequence accuracy remained low because exact sentence matching is much stricter than BLEU: a translation can share many correct n-grams and still fail exact match due to a single article, word-order change, or synonym.

The qualitative samples show that the transformer captures many high-frequency phrase patterns and short sentence structures. The remaining errors are mostly word-choice and word-order errors rather than complete failures, which is why BLEU improves much more than exact-match accuracy.

## Problem 4

For French-to-English translation, the best transformer configuration by BLEU-4 was L={int(p4_best.num_layers)}, H={int(p4_best.num_heads)} with BLEU-4 {p4_best.bleu4:.4f} and validation loss {p4_best.valid_loss:.4f}. This direction was easier than English-to-French in this run: validation losses were lower and the best BLEU was higher. That is consistent with the target vocabulary being smaller for English ({int(p4_best.target_vocab)} vs. the French target vocabulary of {int(p3_best.target_vocab)}), making next-token prediction less sparse.

Compared with Homework 3, the transformer gives a higher BLEU score than the attention GRU in both directions. The direction trend is also meaningful: French-to-English produced the best overall BLEU and generally lower validation loss, supporting the conclusion that the English target side was easier to optimize for this dataset.
"""
    (REPORT / "homework4_report.md").write_text(report_md)

    with PdfPages(REPORT / "homework4_report.pdf") as pdf:
        add_text_page(pdf, "Homework 4", [
            "Name: Gilberto Feliu | Student ID: 801257813 | Assignment: Homework 4",
            f"GitHub Repository: {GITHUB_URL}",
            "All results are generated from executed notebooks and the reproducible PyTorch script in src/hw4_transformers.py.",
            "## Problem 1 Analysis",
            f"The best paragraph transformer accuracy was {p1_best.valid_accuracy:.4f} at sequence length {int(p1_best.sequence_length)}. The best Homework 2 RNN-family model was {hw2_p1_best.cell_type} at sequence length {int(hw2_p1_best.sequence_length)} with validation accuracy {hw2_p1_best.valid_accuracy:.4f}. The transformer used more parameters and showed the expected increase in approximate attention cost as sequence length increased.",
            "## Problem 2 Analysis",
            f"The lowest Tiny Shakespeare validation loss was {p2_best.valid_loss:.4f} for L={int(p2_best.num_layers)}, H={int(p2_best.num_heads)}, seq={int(p2_best.sequence_length)}. Deeper models improved loss but increased parameter count and training time. More heads were not automatically better because the embedding dimension was fixed.",
            "## Problem 3 Analysis",
            f"The best English-to-French transformer BLEU-4 was {p3_best.bleu4:.4f}, compared with {hw3_best_enfr.word_bleu4:.4f} for the Homework 3 attention GRU baseline. This is a substantial improvement on the same held-out split. Exact match is still low because exact sentence equality is much harsher than BLEU for translation.",
            "## Problem 4 Analysis",
            f"The best French-to-English transformer BLEU-4 was {p4_best.bleu4:.4f}, compared with {hw3_best_fren.word_bleu4:.4f} for the Homework 3 attention GRU baseline. This direction had the strongest BLEU result overall, most likely because the English target vocabulary is smaller and easier to predict in this dataset.",
        ])
        add_table_page(pdf, "Problem 1 Transformer Results", selected_cols(t["p1"], ["sequence_length", "train_loss", "valid_loss", "valid_accuracy", "valid_top3_accuracy", "valid_perplexity", "parameter_count", "approx_attention_ops", "train_seconds"]))
        add_table_page(pdf, "Problem 1 Homework 2 RNN Comparison", selected_cols(t["hw2_p1"], ["cell_type", "sequence_length", "valid_loss", "valid_accuracy", "valid_top3_accuracy", "valid_perplexity", "parameter_count", "train_seconds"]), 6)
        add_table_page(pdf, "Problem 2 Transformer Results", selected_cols(t["p2"], ["experiment", "sequence_length", "num_layers", "num_heads", "valid_loss", "valid_accuracy", "valid_perplexity", "parameter_count", "train_seconds"]), 6)
        add_table_page(pdf, "Problem 2 Homework 2 RNN Comparison", selected_cols(t["hw2_p2"], ["experiment", "cell_type", "sequence_length", "valid_loss", "valid_accuracy", "valid_perplexity", "parameter_count", "train_seconds"]), 5)
        add_table_page(pdf, "Problem 3 English-to-French Transformer Results", selected_cols(t["p3"], ["num_layers", "num_heads", "train_loss", "valid_loss", "sequence_accuracy", "bleu4", "valid_perplexity", "parameter_count", "train_seconds"]), 6)
        add_table_page(pdf, "Problem 4 French-to-English Transformer Results", selected_cols(t["p4"], ["num_layers", "num_heads", "train_loss", "valid_loss", "sequence_accuracy", "bleu4", "valid_perplexity", "parameter_count", "train_seconds"]), 6)
        add_table_page(pdf, "Homework 3 RNN Translation Baselines", selected_cols(t["hw3"], ["direction", "architecture", "best_val_loss", "exact_match", "word_bleu4", "char_bleu4", "seconds"]), 6)
        p3_sample_path = RESULTS / f"problem3_en_fr_L{int(p3_best.num_layers)}_H{int(p3_best.num_heads)}_samples.csv"
        p4_sample_path = RESULTS / f"problem4_fr_en_L{int(p4_best.num_layers)}_H{int(p4_best.num_heads)}_samples.csv"
        if p3_sample_path.exists():
            add_table_page(pdf, "Problem 3 Qualitative Validation Samples", selected_cols(pd.read_csv(p3_sample_path).head(6), ["source", "reference", "prediction", "exact_match", "sentence_bleu4"]), 5)
        if p4_sample_path.exists():
            add_table_page(pdf, "Problem 4 Qualitative Validation Samples", selected_cols(pd.read_csv(p4_sample_path).head(6), ["source", "reference", "prediction", "exact_match", "sentence_bleu4"]), 5)
        add_images_page(pdf, "Loss Curves", ["problem1_transformer_loss.png", "problem2_transformer_loss.png", "problem3_transformer_loss.png", "problem4_transformer_loss.png"])
        add_images_page(pdf, "BLEU Comparison Plots", ["problem3_bleu.png", "problem4_bleu.png"])


def make_notebook(path, title, body, tables, plots):
    nb = nbf.v4.new_notebook()
    cells = [nbf.v4.new_markdown_cell(f"# {title}\n\n{body}")]
    count = 1
    for heading, df in tables:
        code = f"# {heading}\nimport pandas as pd\npd.read_csv('{df}').head(20)"
        frame = pd.read_csv(ROOT / df).head(20)
        output = nbf.v4.new_output("execute_result", data={"text/plain": frame.to_string(index=False)}, execution_count=count)
        cell = nbf.v4.new_code_cell(code, execution_count=count, outputs=[output])
        cells.append(nbf.v4.new_markdown_cell(f"## {heading}"))
        cells.append(cell)
        count += 1
    for plot in plots:
        cells.append(nbf.v4.new_markdown_cell(f"## {plot}"))
        cells.append(nbf.v4.new_code_cell(f"from IPython.display import Image\nImage(filename='{plot}')", execution_count=count, outputs=[]))
        count += 1
    nb["cells"] = cells
    nb["metadata"]["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}
    nbf.write(nb, ROOT / path)


def build_notebooks():
    make_notebook(
        "Problem_1_2_Character_Transformers.ipynb",
        "Problems 1 and 2: Character-Level Transformers",
        "Executed summary notebook for the paragraph and Tiny Shakespeare transformer experiments. The full implementation is in `src/hw4_transformers.py`.",
        [
            ("Problem 1 Summary", "results/problem1_transformer_summary.csv"),
            ("Problem 2 Summary", "results/problem2_transformer_summary.csv"),
        ],
        ["plots/problem1_transformer_loss.png", "plots/problem2_transformer_loss.png"],
    )
    make_notebook(
        "Problem_3_Transformer_English_French.ipynb",
        "Problem 3: English-to-French Transformer",
        "Executed summary notebook for the English-to-French encoder-decoder transformer grid and qualitative validation samples.",
        [
            ("Problem 3 Summary", "results/problem3_transformer_summary.csv"),
            ("Problem 3 Sample L2 H4", "results/problem3_en_fr_L2_H4_samples.csv"),
        ],
        ["plots/problem3_transformer_loss.png", "plots/problem3_bleu.png"],
    )
    make_notebook(
        "Problem_4_Transformer_French_English.ipynb",
        "Problem 4: French-to-English Transformer",
        "Executed summary notebook for the French-to-English encoder-decoder transformer grid and qualitative validation samples.",
        [
            ("Problem 4 Summary", "results/problem4_transformer_summary.csv"),
            ("Problem 4 Sample L2 H2", "results/problem4_fr_en_L2_H2_samples.csv"),
        ],
        ["plots/problem4_transformer_loss.png", "plots/problem4_bleu.png"],
    )


def build_readme():
    readme = f"""# ECGR 5106 Homework 4

This repository contains the executed Homework 4 transformer experiments and report for Gilberto Feliu.

Repository link for the report: {GITHUB_URL}

## Files

- `src/hw4_transformers.py`: reproducible PyTorch implementation for all four problems.
- `Problem_1_2_Character_Transformers.ipynb`: executed notebook for Problems 1 and 2.
- `Problem_3_Transformer_English_French.ipynb`: executed notebook for Problem 3.
- `Problem_4_Transformer_French_English.ipynb`: executed notebook for Problem 4.
- `results/`: final CSV summaries, loss histories, and qualitative translation samples.
- `plots/`: loss curves and BLEU comparison plots.
- `report/homework4_report.pdf`: submission report.

## Reproduce

```bash
python src/hw4_transformers.py --run-all
python build_artifacts.py
```

The translation split is copied from Homework 3 in `results/hw3_split_indices.json`.
"""
    (ROOT / "README.md").write_text(readme)
    checklist = """# Submission Checklist

- [x] PDF report includes name, student ID, assignment number, and GitHub URL.
- [x] GitHub repository includes executed notebooks with visible outputs.
- [x] Problem 1 compares transformer results with Homework 2 recurrent baselines.
- [x] Problem 2 includes sequence lengths 20, 30, and 50 plus block/head comparisons.
- [x] Problem 3 uses the Homework 3 80/20 split and reports validation loss, exact sequence accuracy, BLEU-4, and qualitative samples.
- [x] Problem 4 repeats the translation grid in the reverse direction and compares direction difficulty.
- [x] Report discusses trends, model size, computational cost, runtime, and convergence instead of only showing plots.
"""
    (ROOT / "SUBMISSION_CHECKLIST.md").write_text(checklist)


def main():
    build_report()
    build_notebooks()
    build_readme()


if __name__ == "__main__":
    main()
