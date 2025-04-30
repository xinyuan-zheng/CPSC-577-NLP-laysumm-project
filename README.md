# Class project for CPSC 477/577 NLP
Xinyuan Zheng, Iris Wang, Zichen Gong  
## Hybrid Extractive-Abstractive Summarization for Biomedical Articles via LLM Generation for Lay Audience 

Large language models (LLMs) offer new possibilities for generating lay summaries of complex scientific literature. This project explores how to augment LLMs to produce accurate, accessible explanations of domain-specific texts. We frame lay summarization as a two-step process: selecting key content and rewriting it in lay-friendly language. Using the BioLaySumm dataset, we compare hard truncation, TextRank, and supervised RoBERTa-based selection. The sentence selection did not improve performance over simple truncation, suggesting that lay summarization may require broader context and that LLMs are robust to noisy inputs in this setting.

### Environment & dependency
- Python 3.10
- pandas: 2.2.1
- numpy: 1.26.4
- pytorch (torch): 2.6.0+cu124
- networkx: 3.3
- sentence_transformers: 4.0.2
- nltk: 3.8.1
- sklearn: 1.2.2
- vllm: 0.8.3.dev188+gdb9dfcfa

### Data & Evaluation
Overview of the BioLaySumm 2024 Shared Task: https://arxiv.org/pdf/2408.08566

The evaluation model is from https://github.com/TGoldsack1/BioLaySumm2024-evaluation_scripts

### Computational Infra
- We ran the TextRank and RoBERTa models using 1 Nvidia A100 GPU
- We ran the inference of Mistral using 2 Nvidia A100 GPUs (without customized configurations)
