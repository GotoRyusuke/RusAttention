# RusAttention
------------------------------------------------------------
## Introduction
This is a task to find out whether one article (or financial report, whatever) has its focus on the Russian-Ukraine conflict. In other words, we want to know whether one artical is mainly talking abt the war.

## Structure
### I. Main Logic
The main logic is in [`attnToSummary`](./attenToSummary.py) module. To initialise the **summariser**, we first change the `__init__` in that module and have it initialised. After that, we can load the excel table for filing info and do the work. The dictionaries we use are old firends: the [Russian-Ukraine-War dictionary](./rus_dict_lemma.txt) and [Russian Names dictionary](rus_names.txt). In the `AttentionToSummary` class, for every filing we get, we first read all the texts of the items under that fiiling and concatenate them, put it into the summariser, and finally detect whether words/phrases from our dictionaries appear in the summary. The class add a new column to the original excel table named "rus_attn", whose value is "1" if the answer to the previous question is "yes" and "0" otherwise. 

### II. Summarisers
There are 3 summarisers that can be initialised by the `AttentionToSummary` class: [`FinanceSummariser`](./finSum.py), [`LexrankSummariser`](./lexrankSum.py), and [`SpacySummariser`](./spacySum.py).

1. `FinanceSummariser`

The `FinancSummariser` class uses the [PEGASUS model](https://huggingface.co/docs/transformers/model_doc/pegasus) (Zhang et al., 2020) fine-tuned on a novel financial news dataset, which consists of 2K articles from Bloomberg, on topics such as stock, markets, currencies, rate and cryptocurrencies (Passali et al., 2021). Named [human-centered-summarization](https://huggingface.co/human-centered-summarization/financial-summarization-pegasus), hopefully it can help us obtain a summary that better captures the nature of a financial report. The model, however, is very resource-expensive, in the sense that the speed of processing a single filing is at around 30 seconds/filing with a NVIDIA RTX 3090, and the it quickly goes out of memory after processing 10 filings. Considering this problem, we make it a back-up option.

2. `LexrankSummariser`

The `LexrankSummariser` follows the following logic: first, it obtains the sentence embeddings using SBERT (Reimers & Gurevych, 2019), and calculates the distance matrix among sentences as what we do in getting similar words in Word2Vec. Then, it uses LexRank algo (Erkan & Radev, 2004) the 5 most "central" sentences, in the sense that all other sentences are closer to these sentences. Thses central sentences are considered the summary of the aritical. The speed is at medium level, like 2 seconds per filing, and parallel is applicable. Although it needs abt 60 GB memory to process one type of filings, we consider it the most efficient algo and use it in the `AttentionToSummary` class.

3. `SpacySummariser`
This summariser has the simplest logic. It first calculates the word frequencies for all words in one filing, normalises the frequencies, and sums up the word freqs in a sentence to get the "score" for that sentence. Finally, it selects the sentences with the highest scores as the summary of that filing. This algo is a simple statistical method (Luhn, 1958). You can check [`spaCy`](https://spacy.io/api) module for more details abt the algo. Although easy to implement, we find it hard to employ parallel, and the speed is at around 1 second per filing. Conceivably, the easiness is at the price of accuracy; simple statistical method sacrifies the structural features of an article. Therefore, we consider it a back-up summariser.

## Example
The [`test_summary`](./test_summary.py) is an example to use the module. The summariser used in it has been set to be the `LexrankSummariser`.

## References
[1] Zhang, J., Zhao, Y., Saleh, M., & Liu, P. (2020, November). Pegasus: Pre-training with extracted gap-sentences for abstractive summarization. In International Conference on Machine Learning (pp. 11328-11339). PMLR.

[2] Passali, T., Gidiotis, A., Chatzikyriakidis, E., & Tsoumakas, G. (2021, April). Towards human-centered summarization: A case study on financial news. In Proceedings of the First Workshop on Bridging Human–Computer Interaction and Natural Language Processing (pp. 21-27).

[3] Reimers, N., & Gurevych, I. (2019). Sentence-bert: Sentence embeddings using siamese bert-networks. arXiv preprint arXiv:1908.10084.

[4] Erkan, G., & Radev, D. R. (2004). Lexrank: Graph-based lexical centrality as salience in text summarization. Journal of artificial intelligence research, 22, 457-479.

[5] Luhn, H. P. (1958). The automatic creation of literature abstracts. IBM Journal of research and development, 2(2), 159-165.

