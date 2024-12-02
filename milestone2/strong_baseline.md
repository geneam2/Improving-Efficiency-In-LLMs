# Description

Although DistilBERT was trained to capture general-domain knowledge, we observed poor performance when it came to question-answering tasks. So we decided to fine-tune DistilBERT on SQuADv1. We use 8000 training samples evaluated on 2000 validation samples.

# Sample Output

`Example 57300888b2c2fd1400568776:
        Predicted: eight legions in five bases along the Rhine
        Actual:    threat of war
Example 5728455bff5b5019007da078:
        Predicted: CALIPSO satellite
        Actual:    NASA's CALIPSO satellite
Example 57097051ed30961900e84134:
        Predicted: Sky Digital
        Actual:    Sky Active`
We observed improvements in our sample output, where the predicted output nearly matches our actual output.

# Scores

For DistilBERT fine-tuned on SQuADv1 for QA, we achieved

`SQuAD Exact-Match (out of 100): 31
SQuAD F1: 40.29
Training Time: 191.62
Inference Time: 0.01 seconds`
