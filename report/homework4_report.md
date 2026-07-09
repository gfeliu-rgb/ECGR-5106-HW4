# Homework 4

Name: Gilberto Feliu  
Student ID: 801257813  
Assignment: Homework 4  
Repository: https://github.com/Gilbertofeliu/ECGR-5106-HW4

## Experimental Notes

All notebooks are included with executed outputs visible. The code uses PyTorch transformer models, the same Homework 2 character sequence, the provided Tiny Shakespeare file, and the same Homework 3 80/20 train-validation split stored in `results/hw3_split_indices.json`. The CPU run intentionally uses compact models and short training schedules so every requested comparison can be reproduced locally.

## Problem 1

The best transformer validation accuracy on the assigned paragraph was 0.3766 at sequence length 20. Compared with the best Homework 2 RNN-family result (GRU, sequence length 10, validation accuracy 0.5348), this lightweight transformer underperformed in top-1 accuracy but remained competitive in top-3 accuracy. The likely cause is data scale: self-attention has more parameters and less built-in recurrence bias, so it needs more text or more epochs to exploit the longer context.

Training time increased from 10 to 30 characters because causal self-attention scales approximately with sequence_length^2 while the feed-forward layers scale linearly in sequence length. The validation loss did not improve at length 30, which indicates the extra context did not overcome the small-data overfitting/noise penalty.

## Problem 2

For Tiny Shakespeare, the lowest validation loss was 2.1666 from L=4, H=4, sequence length 30. The best Homework 2 recurrent result by validation loss was LSTM with sequence length 30 and validation loss 2.1718. The transformer reached similar character accuracy after only two lightweight epochs, but the recurrent model had a longer training budget and generated more coherent samples.

Depth had the clearest model-size effect: moving from 1 to 4 transformer blocks increased parameters and training time. Four blocks gave the strongest validation loss in this CPU run, suggesting added depth helped representation capacity more than it hurt optimization. Increasing heads from 2 to 4 did not consistently improve validation loss because d_model was fixed at 48, so each head received a smaller subspace.

## Problem 3

For English-to-French translation, the best transformer configuration by BLEU-4 was L=2, H=4 with BLEU-4 0.0067 and validation loss 5.8101. The Homework 3 attention GRU reached word BLEU-4 0.0589; therefore the RNN attention baseline remains stronger under this limited transformer training budget. Exact match was 0 for every model because exact sentence matching is a harsh metric on a small translation set, especially when outputs are short and partially trained.

The qualitative samples show that the transformer often learns frequent tokens before full sentence structure. That behavior matches the high validation perplexity and low BLEU scores: the model is beginning to fit the vocabulary distribution but has not converged enough for fluent sentence-level generation.

## Problem 4

For French-to-English translation, the best transformer configuration by BLEU-4 was L=2, H=2 with BLEU-4 0.0118 and validation loss 5.4851. This direction was easier than English-to-French in this run: validation losses were lower and the best BLEU was higher. That is consistent with the target vocabulary being smaller for English (901 vs. the French target vocabulary of 996), making next-token prediction less sparse.

The comparison with Homework 3 still favors the attention GRU in absolute BLEU, but the transformer direction trend is meaningful: French-to-English produced stronger held-out loss and BLEU under the same split, architecture grid, and training budget.
