## Exploring the Impact of Corpus Diversity on Financial Pretrained Language Models
(EMNLP 2023 findings)

Paper: https://arxiv.org/abs/2310.13312

model repository: https://huggingface.co/HYdsl/FiLM

### Abstract
Over the past few years, various domain-specific pretrained language models (PLMs) have been proposed and have outperformed general-domain PLMs in specialized areas such as biomedical, scientific, and clinical domains. In addition, financial PLMs have been studied because of the high economic impact of financial data analysis. However, we found that financial PLMs were not pretrained on sufficiently diverse financial data. This lack of diverse training data leads to a subpar generalization performance, resulting in general-purpose PLMs, including BERT, often outperforming financial PLMs on many downstream tasks. To address this issue, we collected a broad range of financial corpus and trained the Financial Language Model (FiLM) on these diverse datasets. Our experimental results confirm that FiLM outperforms not only existing financial PLMs but also general domain PLMs. Furthermore, we provide empirical evidence that this improvement can be achieved even for unseen corpus groups.

### **FiLM**(**Fi**nancial **L**anguage **M**odel) Models 🌟
FiLM is a Pre-trained Language Model (PLM) optimized for the Financial domain, built upon a diverse range of Financial domain corpora. Initialized with the RoBERTa-base model, FiLM undergoes further training to achieve performance that surpasses RoBERTa-base in financial domain for the first time.

To train FiLM, we have categorized our Financial Corpus into specific groups and gathered a diverse range of corpora to ensure optimal performance.

We offer two versions of the FiLM model, each tailored for specific use-cases in the Financial domain:

[**FiLM (2.4B): Our Base Model**](https://huggingface.co/HYdsl/FiLM)

This is our foundational model, trained on the entire range of corpora as outlined in the above Corpus table. Ideal for a wide array of financial applications. 📊

[**FiLM (5.5B): Optimized for SEC Filings**](https://huggingface.co/HYdsl/FiLM-SEC)

This model is specialized for handling SEC filings. We expanded the training set by adding 3.1 billion tokens from the SEC filings corpus dataset. The dataset is sourced from EDGAR-CORPUS: Billions of Tokens Make The World Go Round (Loukas et al., ECONLP 2021)

The method to load a tokenizer and a model.
For the FiLM model, you can call 'roberta-base' from the tokenizer.
```python
tokenizer = AutoTokenizer.from_pretrained('roberta-base')
model = AutoModel.from_pretrained('HYdsl/FiLM')
```

<table>
  <tr align="center">
    <td>Groupd</td>
    <td>Name</td>
    <td>Description</td>
    <td># Tokens</td>
  </tr>
  <tr align="center">
    <td align="center" rowspan="4">News</td>
    <td>TRC2</td>
    <td>Collection financial news stories from Reuters</td>
    <td> 227.39 M </td>
  </tr>
  <tr align="center">
    <td>Investing.com</td>
    <td>Stock, options, commodity etc. News article</td>
    <td> 130.88 M </td>
  </tr>
  <tr align="center">
    <td>NYtimes</td>
    <td>Economic articles from the New York Times</td>
    <td> 75.04 M </td>
  </tr>
  <tr align="center">
    <td>EIA</td>
    <td>Commodity related news articles from EIA</td>
    <td> 1.12 M </td>
  </tr>
  <tr align="center">
    <td align="center" colspan="2">SEC filings</td>
    <td>Annual reports(10-K) and quarterly reports(10-Q)</td>
    <td> 307.19 M </td>
  </tr>
  <tr align="center">
    <td align="center" colspan="2">Earnings Call</td>
    <td>Earnings conference call transcripts</td>                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   
    <td> 1.66 B </td>
  </tr>
  <tr align="center">
    <td align="center" rowspan="2">Papers</td>
    <td>ArXiv</td>
    <td>A collection of abstracts of economic research papers</td>
    <td> 42.18 M </td>
  </tr>
  <tr align="center">
    <td>AIHUB</td>
    <td>A collection of Korean economics research papers</td>
    <td> 5.89 M </td>
  </tr>
  <tr align="center">
    <td align="center" rowspan="2">MISC</td>
    <td>Investopedia</td>
    <td>Economic glossary</td>
    <td> 5.33 M </td>
  </tr>
  <tr align="center">
    <td>FinWEB</td>
    <td>Finance, loans, and insurance related articles</td>
    <td> 2.86 M </td>
  </tr>
  <tr align="center">
    <td colspan="3" align="center"> A total of 10 corpora </td>
    <td> 2.4 B </td>
  </tr>
</table>
    
