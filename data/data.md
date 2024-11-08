# Data Description

## Overview

The WikiText language modeling dataset is a collection of over 100 million tokens extracted from the set of verified Good and Featured articles on Wikipedia. It contains two main datasets: wikitext-103 and wikitext-2, each with two variants: raw (for character level work) and non-raw (for word level work). Each variant is already split into train, validation, and test splits stored as parquet files with the number of rows shown in the table below.

| **Name**         | **File Format** | **Train** | **Validation** | **Test** |
| ---------------- | --------------- | --------- | -------------- | -------- |
| wikitext-103-raw | parquet         | 1,801,350 | 3,760          | 4,358    |
| wikitext-103     | parquet         | 1,801,350 | 3,760          | 4,358    |
| wikitext-2-raw   | parquet         | 36,718    | 3,760          | 4,358    |
| wikitext-2       | parquet         | 36,718    | 3,760          | 4,358    |

The raw variant contains the raw tokens whereas the non-raw variant only contains tokens in the wiki.train.tokens, wiki.valid.tokens, and wiki.test.tokens. Out-of-vocabulary words are replaced with the token `UNK`.

## Dataset Link

The full dataset can be accessed here:
[WikiText on Hugging Face](https://huggingface.co/datasets/Salesforce/wikitext)

## Example Data

Here is a snippet from the wikitext-103-raw dataset:

{
"text":
"Senjō no Valkyria 3 : Unrecorded Chronicles ( Japanese : 戦場のヴァルキュリア 3 , lit . Valkyria of the Battlefield 3 )"
}

In wiki-text-103, this would look like:

{
"text":
"' Senjō no Valkyria 3 : <unk> Chronicles ( Japanese : 戦場のヴァルキュリア 3 , lit . Valkyria of the Battlefield 3 )"
}

### Licensing Information

The dataset is available under the [Creative Commons Attribution-ShareAlike License (CC BY-SA 4.0)](https://creativecommons.org/licenses/by-sa/4.0/).

### Citation Information

```
@misc{merity2016pointer,
      title={Pointer Sentinel Mixture Models},
      author={Stephen Merity and Caiming Xiong and James Bradbury and Richard Socher},
      year={2016},
      eprint={1609.07843},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```

### Contributions

Thanks to [@thomwolf](https://github.com/thomwolf), [@lewtun](https://github.com/lewtun), [@patrickvonplaten](https://github.com/patrickvonplaten), [@mariamabarham](https://github.com/mariamabarham) for adding this dataset.
