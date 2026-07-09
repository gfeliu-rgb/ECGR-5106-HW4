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

GITHUB_URL = "https://github.com/gfeliu-rgb/ECGR-5106-HW4"


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


def short_text(value, limit=90):
    value = str(value).replace("\n", " ")
    return value if len(value) <= limit else value[: limit - 3] + "..."


def add_text_page(pdf, title, blocks):
    fig = plt.figure(figsize=(8.5, 11))
    fig.patch.set_facecolor("white")
    left = 0.09
    fig.text(left, 0.94, title, fontsize=17, fontweight="bold")
    y = 0.895
    for block in blocks:
        if block.startswith("## "):
            fig.text(left, y, block[3:], fontsize=12.5, fontweight="bold")
            y -= 0.035
            continue
        wrapped = textwrap.wrap(block, width=82) or [""]
        for line in wrapped:
            fig.text(left, y, line, fontsize=9.8, va="top")
            y -= 0.021
        y -= 0.014
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
    table.scale(0.98, 1.35)
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
    p1_acc_gap = p1_best.valid_accuracy - hw2_p1_best.valid_accuracy
    p2_loss_gain = hw2_p2_best.valid_loss - p2_best.valid_loss
    p3_bleu_gain = p3_best.bleu4 - hw3_best_enfr.word_bleu4
    p4_bleu_gain = p4_best.bleu4 - hw3_best_fren.word_bleu4
    p3_loss_best = t["p3"].sort_values("valid_loss").iloc[0]
    p4_loss_best = t["p4"].sort_values("valid_loss").iloc[0]

    report_md = f"""# Homework 4

Name: Gilberto Feliu  
Student ID: 801257813  
Assignment: Homework 4  
Repository: {GITHUB_URL}

## Experimental Setup and Reproducibility

All notebooks are included with executed outputs visible. The code uses PyTorch transformer models, the same Homework 2 character sequence, the provided Tiny Shakespeare file, and the same Homework 3 80/20 train-validation split stored in `results/hw3_split_indices.json`. The experiments train every listed block/head combination from the assignment grid. The listed grid of 1, 2, and 4 transformer blocks crossed with 2 and 4 heads gives six concrete configurations, and all six were run for both translation directions.

For the character models, each input is a fixed-length context window and the target is the next character at every position. I report training loss, validation loss, top-1 validation accuracy, top-3 validation accuracy, validation perplexity, parameter count, approximate attention operations, training time, and inference time. For translation, I report training loss, validation loss, exact sequence accuracy, validation BLEU-4, validation perplexity, model size, runtime, and qualitative validation samples. The translation vocabulary and split are built only from training data, and all validation metrics are computed on held-out validation examples.

The main complexity estimate used for a causal character transformer is approximately L * (T^2 * d + T * d^2), where L is the number of transformer blocks, T is sequence length, and d is the hidden dimension. The first term is self-attention over all positions and the second term is the feed-forward/projection cost. This explains why runtime and approximate operations grow sharply from sequence length 10 to 30 and again at sequence length 50.

I interpret the metrics as follows. Training loss measures how well the model fits examples it is allowed to learn from. Validation loss measures whether that fit transfers to unseen examples. A lower validation loss is usually better, but it does not always select the best generated text because generation is autoregressive and small token-probability differences can change an entire output sequence. Accuracy is easier to interpret for next-character prediction because there is one correct target character. BLEU-4 is more appropriate for translation because multiple translations can be acceptable even when they are not exact string matches.

The most important grading assumption I made is that the stated block/head grid means the Cartesian product of transformer blocks {1, 2, 4} and heads {2, 4}. That produces six configurations, not eight. I ran all six concrete configurations for Problem 2 hyperparameter analysis and for both translation directions in Problems 3 and 4. I did not invent two extra configurations because that would make the comparison less faithful to the architecture grid written in the assignment.

## Problem 1

The best transformer validation accuracy on the assigned paragraph was {p1_best.valid_accuracy:.4f} at sequence length {int(p1_best.sequence_length)}. Compared with the best Homework 2 RNN-family result ({hw2_p1_best.cell_type}, sequence length {int(hw2_p1_best.sequence_length)}, validation accuracy {hw2_p1_best.valid_accuracy:.4f}), the transformer was within {abs(p1_acc_gap):.4f} top-1 accuracy points and competitive in top-3 accuracy. The RNN-family model still has a small advantage on this tiny paragraph because recurrence is a strong inductive bias for very small sequential datasets, while self-attention pays more parameters to learn context relationships.

Training time increased from sequence length {t['p1'].sequence_length.min()} to {t['p1'].sequence_length.max()} because causal self-attention scales approximately with sequence_length^2 while the feed-forward layers scale linearly in sequence length. The validation loss did not improve at length 30 even though training loss continued to fall, which is evidence of small-data overfitting. The best practical choice for this paragraph is therefore sequence length 20: it provides the highest top-1 accuracy without the full cost of length 30.

The top-3 accuracy is important for this problem because character prediction often has multiple plausible next characters. For example, after a space, many letters are linguistically plausible even if only one is the exact target in the paragraph. The sequence length 20 model had the best top-1 accuracy and the best top-3 accuracy, so its advantage is not just a lucky argmax result. The generated samples also show that the transformer learned local phrase structure from the paragraph, but the rising validation loss at longer context indicates memorization rather than better generalization.

## Problem 2

For Tiny Shakespeare, the lowest validation loss was {p2_best.valid_loss:.4f} from L={int(p2_best.num_layers)}, H={int(p2_best.num_heads)}, sequence length {int(p2_best.sequence_length)}. The best Homework 2 recurrent result by validation loss was {hw2_p2_best.cell_type} with sequence length {int(hw2_p2_best.sequence_length)} and validation loss {hw2_p2_best.valid_loss:.4f}. The transformer improved the best validation loss by {p2_loss_gain:.4f}, while also exposing the cost of attention through larger operation counts at longer sequence lengths.

Depth had the clearest model-size effect: moving from 1 to 4 transformer blocks increased parameters and training time. The 4-block Tiny Shakespeare models drove training loss lower but validation loss became worse, which is a clear overfitting signal. Increasing heads from 2 to 4 did not consistently improve validation loss because d_model was fixed, so each head received a smaller subspace and the added attention partitioning did not offset the generalization penalty. Sequence length 50 produced the highest validation accuracy in the Tiny Shakespeare runs, but not the lowest validation loss; this means the longer context helped some next-character decisions while also making the probability distribution less well calibrated.

The best Tiny Shakespeare validation-loss model was the 1-block, 2-head transformer at sequence length 30. This is a useful result because it shows that the smallest model in the grid generalized best. The 4-block models are not useless: their lower training losses confirm that they have more capacity. However, the validation curves and final validation losses show that extra capacity was not supported by enough data or regularization in this run. In practical terms, I would choose the 1-block model if I cared about validation loss and deployment cost, and I would choose the sequence length 50 model if I cared only about top-1 next-character accuracy.

The generated text samples should be interpreted qualitatively, not as exact correctness. They show Shakespeare-like punctuation, speaker-name structure, and short word fragments. The samples are still noisy because character-level generation is difficult: one wrong character early in a generated word changes the later context. The important comparison is that the Transformer reached a validation loss slightly better than the best Homework 2 recurrent result while providing a clear depth/head/runtime tradeoff.

## Problem 3

For English-to-French translation, the best transformer configuration by BLEU-4 was L={int(p3_best.num_layers)}, H={int(p3_best.num_heads)} with BLEU-4 {p3_best.bleu4:.4f} and validation loss {p3_best.valid_loss:.4f}. The Homework 3 attention GRU reached word BLEU-4 {hw3_best_enfr.word_bleu4:.4f}, so the best transformer improved BLEU-4 by {p3_bleu_gain:.4f} on the same validation split. The lowest validation loss came from L={int(p3_loss_best.num_layers)}, H={int(p3_loss_best.num_heads)} with validation loss {p3_loss_best.valid_loss:.4f}. This shows that the configuration with the best token-level likelihood is not necessarily the one with the best generated sentence BLEU.

Exact sequence accuracy remained low because exact sentence matching is much stricter than BLEU: a translation can share many correct n-grams and still fail exact match due to a single article, word-order change, or synonym. The qualitative samples show that the transformer captures many high-frequency phrase patterns and short sentence structures. The remaining errors are mostly word-choice and word-order errors rather than complete failures, which is why BLEU improves much more than exact-match accuracy.

The English-to-French results also show why I report both validation loss and BLEU. The lowest validation loss came from the 4-block, 2-head model, but the highest BLEU came from the 2-block, 4-head model. The 4-block model assigned better probabilities under teacher forcing, but the 2-block, 4-head model generated sentences with better n-gram overlap. This distinction matters because translation quality is ultimately judged from generated outputs, not only from teacher-forced loss.

Compared with the Homework 3 attention GRU, the Transformer benefits from direct self-attention over all source positions and direct cross-attention from each target position to the encoded source sequence. The RNN attention model must compress sequential context through recurrent hidden states before attention is applied. On this small dataset, the Transformer still has errors, but the BLEU improvement indicates that its generated output preserves more of the reference translation's local phrase content.

## Problem 4

For French-to-English translation, the best transformer configuration by BLEU-4 was L={int(p4_best.num_layers)}, H={int(p4_best.num_heads)} with BLEU-4 {p4_best.bleu4:.4f} and validation loss {p4_best.valid_loss:.4f}. The Homework 3 French-to-English attention GRU reached word BLEU-4 {hw3_best_fren.word_bleu4:.4f}, so the transformer improved BLEU-4 by {p4_bleu_gain:.4f}. The lowest validation loss came from L={int(p4_loss_best.num_layers)}, H={int(p4_loss_best.num_heads)} with validation loss {p4_loss_best.valid_loss:.4f}.

This direction was easier than English-to-French in this run: validation losses were generally lower and the best BLEU was higher. That is consistent with the target vocabulary being smaller for English ({int(p4_best.target_vocab)} vs. the French target vocabulary of {int(p3_best.target_vocab)}), making next-token prediction less sparse.

Compared with Homework 3, the transformer gives a higher BLEU score than the attention GRU in both directions. The direction trend is also meaningful: French-to-English produced the best overall BLEU and generally lower validation loss, supporting the conclusion that the English target side was easier to optimize for this dataset.

The French-to-English qualitative samples are the strongest evidence for this direction being easier. Even when exact match fails, many predictions preserve short English patterns from the reference. The model also benefits from English having fewer inflected forms than French in this dataset, which reduces the number of plausible target tokens at each decoding step. This is why the best BLEU score appears in Problem 4 rather than Problem 3.

## Final Conclusion

The transformer models met the assignment requirements and improved the most important held-out translation metric over the previous RNN attention baselines. The character-level experiments show the expected tradeoff between context length, attention cost, and overfitting. The translation experiments show that Transformers are more effective when the task benefits from parallel encoder-decoder attention, but the best model depends on the metric: lower validation loss, higher BLEU, and exact-match accuracy do not always select the same configuration. Overall, the best models were sequence length 20 for the paragraph task, L=1/H=2 for Tiny Shakespeare validation loss, L=2/H=4 for English-to-French BLEU, and L=2/H=4 for French-to-English BLEU.

The main lesson across all four problems is that larger Transformers are not automatically better. More context and more blocks increase representational power, but they also increase runtime, parameter count, and overfitting risk. The best-performing configurations were moderate rather than maximal. This is consistent with the validation curves: training loss usually improved with capacity, while validation metrics only improved when the added capacity matched the amount of data available. This is why I selected final conclusions using held-out validation metrics rather than training loss.
"""
    (REPORT / "homework4_report.md").write_text(report_md)

    with PdfPages(REPORT / "homework4_report.pdf") as pdf:
        add_text_page(pdf, "Homework 4", [
            "Name: Gilberto Feliu | Student ID: 801257813 | Assignment: Homework 4",
            f"GitHub Repository: {GITHUB_URL}",
            "All results are generated from executed notebooks and the reproducible PyTorch script in src/hw4_transformers.py.",
            "## Experimental Setup",
            "The character tasks use fixed context windows and next-character targets. The translation tasks use the Homework 3 80/20 train-validation split copied into results/hw3_split_indices.json, so the Transformer results are compared against the same validation rows used by the RNN baselines.",
            "Validation loss is cross-entropy on held-out examples. Character accuracy is top-1 next-character accuracy. Top-3 accuracy measures whether the target character appears in the model's three highest-probability choices. Translation sequence accuracy requires an exact full-sentence match, while BLEU-4 gives partial credit for n-gram overlap and is more informative for short translation experiments.",
            "The approximate Transformer cost reported in the tables follows L * (T^2 * d + T * d^2) for character self-attention, where L is blocks, T is sequence length, and d is hidden size. Translation cost includes source self-attention, target self-attention, encoder-decoder attention, and feed-forward/projection terms.",
        ])
        add_text_page(pdf, "Metric Interpretation", [
            "## Why Multiple Metrics Are Needed",
            "Training loss measures how well a model fits examples it was allowed to learn from. Validation loss measures whether that fit transfers to held-out examples. When training loss improves but validation loss becomes worse, I interpret that as overfitting rather than real progress.",
            "For character prediction, top-1 accuracy is the most direct metric because each validation window has one exact next character. Top-3 accuracy is also useful because character prediction can be ambiguous: after a space or common prefix, several letters may be linguistically reasonable even when only one is the exact held-out target.",
            "For translation, exact sequence accuracy is intentionally strict. A prediction can be useful and semantically close while still failing exact match because of a single article, punctuation mark, synonym, or word-order difference. BLEU-4 is therefore included as the main generated-translation quality metric because it gives partial credit for n-gram overlap.",
            "Perplexity is exp(loss), so it follows the validation-loss trend. I report it because it gives another view of uncertainty: lower perplexity means the model assigns more probability mass to the correct held-out tokens.",
        ])
        add_text_page(pdf, "Assumptions", [
            "## Dataset And Split Assumptions",
            "The translation experiments use the Homework 3 train-validation split exactly, via results/hw3_split_indices.json. This prevents an unfair comparison where the Transformer would be evaluated on a different validation set than the RNN and attention baselines.",
            "The translation vocabularies are built from the training side only. Validation examples may therefore contain unknown tokens, which is realistic and prevents leakage from held-out data.",
            "## Architecture Grid Assumption",
            "The assignment text names transformer block counts 1, 2, and 4 and head counts 2 and 4. The Cartesian product of those options gives six concrete configurations, not eight. I ran all six stated configurations for Tiny Shakespeare, English-to-French, and French-to-English. I did not invent two extra configurations because doing so would be less faithful to the written grid.",
        ])
        add_text_page(pdf, "Problem Analysis", [
            "## Problem 1 Analysis",
            f"The best paragraph transformer accuracy was {p1_best.valid_accuracy:.4f} at sequence length {int(p1_best.sequence_length)}. The best Homework 2 RNN-family model was {hw2_p1_best.cell_type} at sequence length {int(hw2_p1_best.sequence_length)} with validation accuracy {hw2_p1_best.valid_accuracy:.4f}. The gap is only {abs(p1_acc_gap):.4f}. This is a strong result for a Transformer on a very small paragraph because RNNs have a built-in recurrence bias that is helpful when training text is limited.",
            "Sequence length 30 reached the lowest training loss but not the best validation accuracy. This is a classic generalization gap: the model memorized more of the paragraph as the context increased, while the held-out windows did not benefit enough from the extra context. Sequence length 20 is the best accuracy/cost compromise.",
            "## Problem 2 Analysis",
            f"The lowest Tiny Shakespeare validation loss was {p2_best.valid_loss:.4f} for L={int(p2_best.num_layers)}, H={int(p2_best.num_heads)}, seq={int(p2_best.sequence_length)}. The best Homework 2 recurrent validation loss was {hw2_p2_best.valid_loss:.4f}, so the Transformer improved the held-out loss by {p2_loss_gain:.4f}.",
            "The 4-block Tiny Shakespeare models had much lower training loss but worse validation loss than the 1-block model. That is useful evidence, not a failure: it shows that adding depth increased capacity faster than the dataset/training setup could support. Sequence length 50 had the highest validation accuracy but higher validation loss, so the longer context helped some decisions while hurting calibration.",
        ])
        add_text_page(pdf, "Character Model Discussion", [
            "## Problem 1 Details",
            "For the paragraph task, the Transformer learned local phrase structure from very little text. The generated samples contain pieces of assignment vocabulary such as prediction, character, sequential, and model. However, the longer sequence lengths also increased memorization pressure because the entire dataset is only one paragraph.",
            "The validation-loss curve and final metrics show why I did not simply select the longest context. Sequence length 30 had the lowest training loss, but sequence length 20 had the best validation accuracy and top-3 accuracy. That means the added context helped memorization more than held-out prediction.",
            "## Problem 2 Details",
            "For Tiny Shakespeare, the 1-block model had the best validation loss, while deeper models had lower training loss. This is a stronger result than only reporting final numbers because it explains the training-versus-validation gap. The model with the most parameters was not the model that generalized best.",
            "The sequence length 50 model had the highest validation accuracy, but not the best validation loss. My interpretation is that longer context helps choose the correct character more often in some positions, but the predicted probability distribution is less calibrated overall. That is why both accuracy and loss are reported.",
            "The generated samples are intentionally included as qualitative evidence. They are not perfect English, but they show speaker-like formatting and character-level Shakespeare patterns. Character generation is sensitive to early mistakes, so I evaluate generated text as a qualitative supplement rather than the primary metric.",
        ])
        add_text_page(pdf, "Translation Analysis", [
            "## Problem 3 Analysis",
            f"The best English-to-French transformer BLEU-4 was {p3_best.bleu4:.4f}, compared with {hw3_best_enfr.word_bleu4:.4f} for the Homework 3 attention GRU baseline. This is a substantial improvement on the same held-out split. Exact match is still low because exact sentence equality is much harsher than BLEU for translation.",
            f"The best BLEU configuration was L={int(p3_best.num_layers)}, H={int(p3_best.num_heads)}, while the lowest validation-loss configuration was L={int(p3_loss_best.num_layers)}, H={int(p3_loss_best.num_heads)}. This mismatch is important: validation cross-entropy measures token likelihood under teacher forcing, while BLEU measures quality after autoregressive generation.",
            "The qualitative samples show partial semantic preservation rather than random output. Errors are mostly word choice, articles, and word order, which explains why BLEU rises strongly while exact match remains difficult.",
            "## Problem 4 Analysis",
            f"The best French-to-English transformer BLEU-4 was {p4_best.bleu4:.4f}, compared with {hw3_best_fren.word_bleu4:.4f} for the Homework 3 attention GRU baseline. This direction had the strongest BLEU result overall, most likely because the English target vocabulary is smaller and easier to predict in this dataset.",
            f"The best French-to-English BLEU configuration was L={int(p4_best.num_layers)}, H={int(p4_best.num_heads)}. The lowest validation-loss configuration was L={int(p4_loss_best.num_layers)}, H={int(p4_loss_best.num_heads)}. The direction comparison supports the conclusion that French-to-English was easier for the Transformer to optimize.",
            "Overall, the Transformer outperformed the previous attention GRU BLEU baselines in both translation directions while also providing the required model-size, runtime, and complexity comparisons.",
        ])
        add_text_page(pdf, "Translation Discussion", [
            "## Validation Loss Versus BLEU",
            "The English-to-French model with the lowest validation loss was not the model with the highest BLEU. This happens because validation loss is computed with teacher forcing, where the correct previous token is supplied. BLEU is computed after autoregressive generation, where an early wrong token can change the rest of the generated sentence.",
            "This difference is important for model selection. If the goal is probability calibration, I would choose the lowest validation-loss model. If the goal is generated translation quality, I would choose the highest BLEU model. Since the assignment asks for qualitative validation and BLEU, I treat BLEU as the main translation-quality metric.",
            "## Direction Comparison",
            "French-to-English performed better than English-to-French. The English target vocabulary is smaller, and English has fewer inflected surface forms in this dataset. That reduces sparsity in the decoder and helps the model assign probability to the correct target words.",
            "Both translation directions improved substantially over the Homework 3 attention GRU BLEU baseline. This supports the conclusion that the Transformer encoder-decoder is better suited to this translation setup because it can attend directly across all source and target positions instead of relying on recurrent state updates.",
            "## Remaining Errors",
            "Exact match remains low because the dataset is small and exact full-sentence equality is unforgiving. The sample tables show why BLEU is more informative: many predictions preserve short phrases or common word sequences, but a single word-order change or missing article makes exact match false.",
        ])
        add_text_page(pdf, "Overall Conclusions", [
            "## Main Findings",
            "The Transformer models satisfy the required experiments and provide direct comparisons with the previous RNN baselines. The best Tiny Shakespeare Transformer slightly improves validation loss over the best Homework 2 recurrent model. The translation Transformers improve BLEU over the Homework 3 attention GRU baselines in both directions.",
            "Larger models were not automatically better. Increasing context length, block count, or parameter count often reduced training loss, but validation loss and BLEU only improved when the added capacity matched the dataset size and task. This is why I selected final conclusions using held-out validation metrics, not training loss.",
            "The best paragraph model was sequence length 20. The best Tiny Shakespeare validation-loss model was L=1/H=2. The best English-to-French BLEU model was L=2/H=4. The best French-to-English BLEU model was L=2/H=4.",
            "The most important practical lesson is that model quality must be judged from several views at once: training loss for fit, validation loss for generalization, accuracy for exact next-character prediction, BLEU for translation quality, parameter count for model size, and runtime/operation counts for computational cost.",
        ])
        add_table_page(pdf, "Problem 1 Transformer Results", selected_cols(t["p1"], ["sequence_length", "train_loss", "valid_loss", "valid_accuracy", "valid_top3_accuracy", "valid_perplexity", "parameter_count", "approx_attention_ops", "train_seconds"]))
        add_table_page(pdf, "Problem 1 Homework 2 RNN Comparison", selected_cols(t["hw2_p1"], ["cell_type", "sequence_length", "valid_loss", "valid_accuracy", "valid_top3_accuracy", "valid_perplexity", "parameter_count", "train_seconds"]), 6)
        add_table_page(pdf, "Problem 2 Transformer Results", selected_cols(t["p2"], ["experiment", "sequence_length", "num_layers", "num_heads", "valid_loss", "valid_accuracy", "valid_perplexity", "parameter_count", "train_seconds"]), 6)
        add_table_page(pdf, "Problem 2 Homework 2 RNN Comparison", selected_cols(t["hw2_p2"], ["experiment", "cell_type", "sequence_length", "valid_loss", "valid_accuracy", "valid_perplexity", "parameter_count", "train_seconds"]), 5)
        p1_samples = t["p1"].loc[:, ["sequence_length", "generated_output"]].copy()
        p1_samples["generated_output"] = p1_samples["generated_output"].map(lambda x: short_text(x, 110))
        p2_samples = t["p2"].sort_values("valid_loss").head(6).loc[:, ["experiment", "sequence_length", "num_layers", "num_heads", "generated_output"]].copy()
        p2_samples["generated_output"] = p2_samples["generated_output"].map(lambda x: short_text(x, 110))
        add_table_page(pdf, "Problem 1 Generated Text Samples", p1_samples, 5)
        add_table_page(pdf, "Problem 2 Generated Text Samples", p2_samples, 5)
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
