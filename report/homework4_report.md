# Homework 4

Name: Gilberto Feliu  
Student ID: 801257813  
Assignment: Homework 4  
Repository: https://github.com/gfeliu-rgb/ECGR-5106-HW4

## Experimental Setup and Reproducibility

All notebooks are included with executed outputs visible. The code uses PyTorch transformer models, the same Homework 2 character sequence, the provided Tiny Shakespeare file, and the same Homework 3 80/20 train-validation split stored in `results/hw3_split_indices.json`. The experiments train every listed block/head combination from the assignment grid. The listed grid of 1, 2, and 4 transformer blocks crossed with 2 and 4 heads gives six concrete configurations, and all six were run for both translation directions.

For the character models, each input is a fixed-length context window and the target is the next character at every position. I report training loss, validation loss, top-1 validation accuracy, top-3 validation accuracy, validation perplexity, parameter count, approximate attention operations, training time, and inference time. For translation, I report training loss, validation loss, exact sequence accuracy, validation BLEU-4, validation perplexity, model size, runtime, and qualitative validation samples. The translation vocabulary and split are built only from training data, and all validation metrics are computed on held-out validation examples.

The main complexity estimate used for a causal character transformer is approximately L * (T^2 * d + T * d^2), where L is the number of transformer blocks, T is sequence length, and d is the hidden dimension. The first term is self-attention over all positions and the second term is the feed-forward/projection cost. This explains why runtime and approximate operations grow sharply from sequence length 10 to 30 and again at sequence length 50.

## Problem 1

The best transformer validation accuracy on the assigned paragraph was 0.5145 at sequence length 20. Compared with the best Homework 2 RNN-family result (GRU, sequence length 10, validation accuracy 0.5348), the transformer was within 0.0203 top-1 accuracy points and competitive in top-3 accuracy. The RNN-family model still has a small advantage on this tiny paragraph because recurrence is a strong inductive bias for very small sequential datasets, while self-attention pays more parameters to learn context relationships.

Training time increased from sequence length 10 to 30 because causal self-attention scales approximately with sequence_length^2 while the feed-forward layers scale linearly in sequence length. The validation loss did not improve at length 30 even though training loss continued to fall, which is evidence of small-data overfitting. The best practical choice for this paragraph is therefore sequence length 20: it provides the highest top-1 accuracy without the full cost of length 30.

## Problem 2

For Tiny Shakespeare, the lowest validation loss was 2.1635 from L=1, H=2, sequence length 30. The best Homework 2 recurrent result by validation loss was LSTM with sequence length 30 and validation loss 2.1718. The transformer improved the best validation loss by 0.0083, while also exposing the cost of attention through larger operation counts at longer sequence lengths.

Depth had the clearest model-size effect: moving from 1 to 4 transformer blocks increased parameters and training time. The 4-block Tiny Shakespeare models drove training loss lower but validation loss became worse, which is a clear overfitting signal. Increasing heads from 2 to 4 did not consistently improve validation loss because d_model was fixed, so each head received a smaller subspace and the added attention partitioning did not offset the generalization penalty. Sequence length 50 produced the highest validation accuracy in the Tiny Shakespeare runs, but not the lowest validation loss; this means the longer context helped some next-character decisions while also making the probability distribution less well calibrated.

## Problem 3

For English-to-French translation, the best transformer configuration by BLEU-4 was L=2, H=4 with BLEU-4 0.2539 and validation loss 5.1019. The Homework 3 attention GRU reached word BLEU-4 0.0589, so the best transformer improved BLEU-4 by 0.1950 on the same validation split. The lowest validation loss came from L=4, H=2 with validation loss 4.7346. This shows that the configuration with the best token-level likelihood is not necessarily the one with the best generated sentence BLEU.

Exact sequence accuracy remained low because exact sentence matching is much stricter than BLEU: a translation can share many correct n-grams and still fail exact match due to a single article, word-order change, or synonym. The qualitative samples show that the transformer captures many high-frequency phrase patterns and short sentence structures. The remaining errors are mostly word-choice and word-order errors rather than complete failures, which is why BLEU improves much more than exact-match accuracy.

## Problem 4

For French-to-English translation, the best transformer configuration by BLEU-4 was L=2, H=4 with BLEU-4 0.2927 and validation loss 4.4974. The Homework 3 French-to-English attention GRU reached word BLEU-4 0.0771, so the transformer improved BLEU-4 by 0.2156. The lowest validation loss came from L=4, H=4 with validation loss 4.4066.

This direction was easier than English-to-French in this run: validation losses were generally lower and the best BLEU was higher. That is consistent with the target vocabulary being smaller for English (901 vs. the French target vocabulary of 996), making next-token prediction less sparse.

Compared with Homework 3, the transformer gives a higher BLEU score than the attention GRU in both directions. The direction trend is also meaningful: French-to-English produced the best overall BLEU and generally lower validation loss, supporting the conclusion that the English target side was easier to optimize for this dataset.

## Final Conclusion

The transformer models met the assignment requirements and improved the most important held-out translation metric over the previous RNN attention baselines. The character-level experiments show the expected tradeoff between context length, attention cost, and overfitting. The translation experiments show that Transformers are more effective when the task benefits from parallel encoder-decoder attention, but the best model depends on the metric: lower validation loss, higher BLEU, and exact-match accuracy do not always select the same configuration. Overall, the best models were sequence length 20 for the paragraph task, L=1/H=2 for Tiny Shakespeare validation loss, L=2/H=4 for English-to-French BLEU, and L=2/H=4 for French-to-English BLEU.
