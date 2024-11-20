# Description

Our simple baseline is a Question-Answering task using the pre-trained model, DistilBERT. We chose it for its lightweight capabilities as it was distilled from BERT with natural language understanding tasks in mind, making it suitable for our compute-constrained environment. We load the SQuAD dataset, model, and tokenizer, process each example, run it through our model, and then evaluate on 200 question-answer pairs. From there, we are able to get predicted start and end indices of our answers, and construct a predicted answer.

# Sample Output

``
Example 56e1011ecd28a01900c6740c:
Predicted: On 7 January 1900, Tesla left Colorado Springs.[
Actual: His lab was torn down

Example 56e74faf00c9c71400d76f95:
Predicted: of Public
Actual: allegations of professional misconduct

Example 572687e1dd62a815002e8854:
Predicted:
Actual: Newcastle Diamonds  
``

# Scores

For DistilBERT, we achieved
`SQuAD Exact-Match (out of 100): 0
SQuAD F1: 6.68
Our Exact-Match (out of 100):
Our F1: `

# Resources:

https://arxiv.org/abs/1909.10351
