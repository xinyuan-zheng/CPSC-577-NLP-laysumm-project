# Class project for CPSC 477/577 NLP
Xinyuan Zheng, Iris Wang, Zichen Gong  
## Lay summarization for biomedical literature  

The rise of large language models (LLMs) has opened up new ways to process and understand long-form text, making vast amounts of knowledge more accessible. Scientific literature, financial reports, and legal documents often contain critical insights, but their complexity poses a barrier to lay audiences. Bridging this gap requires effective summarization techniques that not only condense information but also translate domain-specific language into lay-friendly interpretations. In this project, we propose to answer: How can LLMs be fine-tuned to generate high-quality lay summaries of domain-specific texts while preserving accuracy and relevance?

Existing summarization work primarily focuses on producing concise, abstractive summaries within the same domain. For example, in finance, earnings call transcripts are distilled into financial news bullet points. In the scientific domain, abstractive and extractive methods are used to generate structured abstracts from research papers. These approaches are effective in condensing information but are designed with domain experts in mind, assuming familiarity with specialized terminology and concepts. To address this gap, we propose to develop an LLM-based framework for lay summarization by leveraging technical-lay twin data and fine-tuning models to generate audience-aware summaries that adjust complexity while preserving accuracy and relevance.



Overview of the BioLaySumm 2024 Shared Task on the Lay Summarization of Biomedical Research Articles: https://arxiv.org/pdf/2408.08566


UIUC_BioNLP at BioLaySumm: An Extract-then-Summarize Approach Augmented with Wikipedia Knowledge for Biomedical Lay Summarization: https://aclanthology.org/2024.bionlp-1.11.pdf