# ECGR 5106 Homework 4

This repository contains the executed Homework 4 transformer experiments and report for Gilberto Feliu.

Repository link for the report: https://github.com/Gilbertofeliu/ECGR-5106-HW4

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
