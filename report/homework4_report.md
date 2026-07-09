# Homework 4

Name: Gilberto Feliu  
Student ID: 801257813  
Assignment: Homework 4  
Repository: https://github.com/gfeliu-rgb/ECGR-5106-HW4

## Experimental Notes

All notebooks are included with executed outputs visible. The code uses PyTorch transformer models, the same Homework 2 character sequence, the provided Tiny Shakespeare file, and the same Homework 3 80/20 train-validation split stored in `results/hw3_split_indices.json`. The experiments train every listed block/head combination from the assignment grid. The listed grid of 1, 2, and 4 transformer blocks crossed with 2 and 4 heads gives six concrete configurations, and all six were run for both translation directions.

## Problem 1

The best transformer validation accuracy on the assigned paragraph was 0.5145 at sequence length 20. Compared with the best Homework 2 RNN-family result (GRU, sequence length 10, validation accuracy 0.5348), the transformer was close in top-1 accuracy and competitive in top-3 accuracy. The RNN-family model still has an advantage on this tiny paragraph because recurrence is a strong inductive bias for very small sequential datasets, while self-attention pays more parameters to learn context relationships.

Training time increased from 10 to 30 characters because causal self-attention scales approximately with sequence_length^2 while the feed-forward layers scale linearly in sequence length. The validation loss did not improve at length 30, which indicates the extra context did not overcome the small-data overfitting/noise penalty.

## Problem 2

For Tiny Shakespeare, the lowest validation loss was 2.1635 from L=1, H=2, sequence length 30. The best Homework 2 recurrent result by validation loss was LSTM with sequence length 30 and validation loss 2.1718. The transformer therefore slightly improved the best validation-loss result while also exposing the cost of attention through larger operation counts at longer sequence lengths.

Depth had the clearest model-size effect: moving from 1 to 4 transformer blocks increased parameters and training time. The 4-block Tiny Shakespeare models drove training loss lower but validation loss became worse, which is a clear overfitting signal. Increasing heads from 2 to 4 did not consistently improve validation loss because d_model was fixed, so each head received a smaller subspace and the added attention partitioning did not offset the generalization penalty.

## Problem 3

For English-to-French translation, the best transformer configuration by BLEU-4 was L=2, H=4 with BLEU-4 0.2539 and validation loss 5.1019. The Homework 3 attention GRU reached word BLEU-4 0.0589, so the best transformer substantially improved the BLEU score on the same validation split. Exact sequence accuracy remained low because exact sentence matching is much stricter than BLEU: a translation can share many correct n-grams and still fail exact match due to a single article, word-order change, or synonym.

The qualitative samples show that the transformer captures many high-frequency phrase patterns and short sentence structures. The remaining errors are mostly word-choice and word-order errors rather than complete failures, which is why BLEU improves much more than exact-match accuracy.

## Problem 4

For French-to-English translation, the best transformer configuration by BLEU-4 was L=2, H=4 with BLEU-4 0.2927 and validation loss 4.4974. This direction was easier than English-to-French in this run: validation losses were lower and the best BLEU was higher. That is consistent with the target vocabulary being smaller for English (901 vs. the French target vocabulary of 996), making next-token prediction less sparse.

Compared with Homework 3, the transformer gives a higher BLEU score than the attention GRU in both directions. The direction trend is also meaningful: French-to-English produced the best overall BLEU and generally lower validation loss, supporting the conclusion that the English target side was easier to optimize for this dataset.
