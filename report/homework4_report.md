# Homework 4

Name: Gilberto Feliu  
Student ID: 801257813  
Assignment: Homework 4  
Repository: https://github.com/gfeliu-rgb/ECGR-5106-HW4

## Experimental Setup and Reproducibility

All notebooks are included with executed outputs visible. The code uses PyTorch transformer models, the same Homework 2 character sequence, the provided Tiny Shakespeare file, and the same Homework 3 80/20 train-validation split stored in `results/hw3_split_indices.json`. The experiments train every listed block/head combination from the assignment grid. The listed grid of 1, 2, and 4 transformer blocks crossed with 2 and 4 heads gives six concrete configurations, and all six were run for both translation directions.

For the character models, each input is a fixed-length context window and the target is the next character at every position. I report training loss, validation loss, top-1 validation accuracy, top-3 validation accuracy, validation perplexity, parameter count, approximate attention operations, training time, and inference time. For translation, I report training loss, validation loss, exact sequence accuracy, validation BLEU-4, validation perplexity, model size, runtime, and qualitative validation samples. The translation vocabulary and split are built only from training data, and all validation metrics are computed on held-out validation examples.

The main complexity estimate used for a causal character transformer is approximately L * (T^2 * d + T * d^2), where L is the number of transformer blocks, T is sequence length, and d is the hidden dimension. The first term is self-attention over all positions and the second term is the feed-forward/projection cost. This explains why runtime and approximate operations grow sharply from sequence length 10 to 30 and again at sequence length 50.

I interpret the metrics as follows. Training loss measures how well the model fits examples it is allowed to learn from. Validation loss measures whether that fit transfers to unseen examples. A lower validation loss is usually better, but it does not always select the best generated text because generation is autoregressive and small token-probability differences can change an entire output sequence. Accuracy is easier to interpret for next-character prediction because there is one correct target character. BLEU-4 is more appropriate for translation because multiple translations can be acceptable even when they are not exact string matches.

The most important grading assumption I made is that the stated block/head grid means the Cartesian product of transformer blocks (1, 2, 4) and heads (2, 4). That produces six configurations, not eight. I ran all six concrete configurations for Problem 2 hyperparameter analysis and for both translation directions in Problems 3 and 4. I did not invent two extra configurations because that would make the comparison less faithful to the architecture grid written in the assignment.

## Problem 1

The best transformer validation accuracy on the assigned paragraph was 0.5145 at sequence length 20. Compared with the best Homework 2 RNN-family result (GRU, sequence length 10, validation accuracy 0.5348), the transformer was within 0.0203 top-1 accuracy points and competitive in top-3 accuracy. The RNN-family model still has a small advantage on this tiny paragraph because recurrence is a strong inductive bias for very small sequential datasets, while self-attention pays more parameters to learn context relationships.

Training time increased from sequence length 10 to 30 because causal self-attention scales approximately with sequence_length^2 while the feed-forward layers scale linearly in sequence length. The validation loss did not improve at length 30 even though training loss continued to fall, which is evidence of small-data overfitting. The best practical choice for this paragraph is therefore sequence length 20: it provides the highest top-1 accuracy without the full cost of length 30.

The top-3 accuracy is important for this problem because character prediction often has multiple plausible next characters. For example, after a space, many letters are linguistically plausible even if only one is the exact target in the paragraph. The sequence length 20 model had the best top-1 accuracy and the best top-3 accuracy, so its advantage is not just a lucky argmax result. The generated samples also show that the transformer learned local phrase structure from the paragraph, but the rising validation loss at longer context indicates memorization rather than better generalization.

## Problem 2

For Tiny Shakespeare, the lowest validation loss was 2.1635 from L=1, H=2, sequence length 30. The best Homework 2 recurrent result by validation loss was LSTM with sequence length 30 and validation loss 2.1718. The transformer improved the best validation loss by 0.0083, while also exposing the cost of attention through larger operation counts at longer sequence lengths.

Depth had the clearest model-size effect: moving from 1 to 4 transformer blocks increased parameters and training time. The 4-block Tiny Shakespeare models drove training loss lower but validation loss became worse, which is a clear overfitting signal. Increasing heads from 2 to 4 did not consistently improve validation loss because d_model was fixed, so each head received a smaller subspace and the added attention partitioning did not offset the generalization penalty. Sequence length 50 produced the highest validation accuracy in the Tiny Shakespeare runs, but not the lowest validation loss; this means the longer context helped some next-character decisions while also making the probability distribution less well calibrated.

The best Tiny Shakespeare validation-loss model was the 1-block, 2-head transformer at sequence length 30. This is a useful result because it shows that the smallest model in the grid generalized best. The 4-block models are not useless: their lower training losses confirm that they have more capacity. However, the validation curves and final validation losses show that extra capacity was not supported by enough data or regularization in this run. In practical terms, I would choose the 1-block model if I cared about validation loss and deployment cost, and I would choose the sequence length 50 model if I cared only about top-1 next-character accuracy.

The generated text samples should be interpreted qualitatively, not as exact correctness. They show Shakespeare-like punctuation, speaker-name structure, and short word fragments. The samples are still noisy because character-level generation is difficult: one wrong character early in a generated word changes the later context. The important comparison is that the Transformer reached a validation loss slightly better than the best Homework 2 recurrent result while providing a clear depth/head/runtime tradeoff.

## Problem 3

For English-to-French translation, the best transformer configuration by BLEU-4 was L=2, H=4 with BLEU-4 0.2539 and validation loss 5.1019. The Homework 3 attention GRU reached word BLEU-4 0.0589, so the best transformer improved BLEU-4 by 0.1950 on the same validation split. The lowest validation loss came from L=4, H=2 with validation loss 4.7346. This shows that the configuration with the best token-level likelihood is not necessarily the one with the best generated sentence BLEU.

Exact sequence accuracy remained low because exact sentence matching is much stricter than BLEU: a translation can share many correct n-grams and still fail exact match due to a single article, word-order change, or synonym. The qualitative samples show that the transformer captures many high-frequency phrase patterns and short sentence structures. The remaining errors are mostly word-choice and word-order errors rather than complete failures, which is why BLEU improves much more than exact-match accuracy.

The English-to-French results also show why I report both validation loss and BLEU. The lowest validation loss came from the 4-block, 2-head model, but the highest BLEU came from the 2-block, 4-head model. The 4-block model assigned better probabilities under teacher forcing, but the 2-block, 4-head model generated sentences with better n-gram overlap. This distinction matters because translation quality is ultimately judged from generated outputs, not only from teacher-forced loss.

Compared with the Homework 3 attention GRU, the Transformer benefits from direct self-attention over all source positions and direct cross-attention from each target position to the encoded source sequence. The RNN attention model must compress sequential context through recurrent hidden states before attention is applied. On this small dataset, the Transformer still has errors, but the BLEU improvement indicates that its generated output preserves more of the reference translation's local phrase content.

## Problem 4

For French-to-English translation, the best transformer configuration by BLEU-4 was L=2, H=4 with BLEU-4 0.2927 and validation loss 4.4974. The Homework 3 French-to-English attention GRU reached word BLEU-4 0.0771, so the transformer improved BLEU-4 by 0.2156. The lowest validation loss came from L=4, H=4 with validation loss 4.4066.

This direction was easier than English-to-French in this run: validation losses were generally lower and the best BLEU was higher. That is consistent with the target vocabulary being smaller for English (901 vs. the French target vocabulary of 996), making next-token prediction less sparse.

Compared with Homework 3, the transformer gives a higher BLEU score than the attention GRU in both directions. The direction trend is also meaningful: French-to-English produced the best overall BLEU and generally lower validation loss, supporting the conclusion that the English target side was easier to optimize for this dataset.

The French-to-English qualitative samples are the strongest evidence for this direction being easier. Even when exact match fails, many predictions preserve short English patterns from the reference. The model also benefits from English having fewer inflected forms than French in this dataset, which reduces the number of plausible target tokens at each decoding step. This is why the best BLEU score appears in Problem 4 rather than Problem 3.

## Final Conclusion

The transformer models met the assignment requirements and improved the most important held-out translation metric over the previous RNN attention baselines. The character-level experiments show the expected tradeoff between context length, attention cost, and overfitting. The translation experiments show that Transformers are more effective when the task benefits from parallel encoder-decoder attention, but the best model depends on the metric: lower validation loss, higher BLEU, and exact-match accuracy do not always select the same configuration. Overall, the best models were sequence length 20 for the paragraph task, L=1/H=2 for Tiny Shakespeare validation loss, L=2/H=4 for English-to-French BLEU, and L=2/H=4 for French-to-English BLEU.

The main lesson across all four problems is that larger Transformers are not automatically better. More context and more blocks increase representational power, but they also increase runtime, parameter count, and overfitting risk. The best-performing configurations were moderate rather than maximal. This is consistent with the validation curves: training loss usually improved with capacity, while validation metrics only improved when the added capacity matched the amount of data available. This is why I selected final conclusions using held-out validation metrics rather than training loss.
