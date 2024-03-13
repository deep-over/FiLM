## Exploring the Impact of Corpus Diversity on Financial Pretrained Language Models
(EMNLP 2023 findings)

Paper: https://arxiv.org/abs/2310.13312

model repository: https://huggingface.co/HYdsl/FiLM

### Abstract
Over the past few years, various domain-specific pretrained language models (PLMs) have been proposed and have outperformed general-domain PLMs in specialized areas such as biomedical, scientific, and clinical domains. In addition, financial PLMs have been studied because of the high economic impact of financial data analysis. However, we found that financial PLMs were not pretrained on sufficiently diverse financial data. This lack of diverse training data leads to a subpar generalization performance, resulting in general-purpose PLMs, including BERT, often outperforming financial PLMs on many downstream tasks. To address this issue, we collected a broad range of financial corpus and trained the Financial Language Model (FiLM) on these diverse datasets. Our experimental results confirm that FiLM outperforms not only existing financial PLMs but also general domain PLMs. Furthermore, we provide empirical evidence that this improvement can be achieved even for unseen corpus groups.

### **FiLM**(**Fi**nancial **L**anguage **M**odel) Models 🌟
FiLM is a Pre-trained Language Model (PLM) optimized for the Financial domain, built upon a diverse range of Financial domain corpora. Initialized with the RoBERTa-base model, FiLM undergoes further training to achieve performance that surpasses RoBERTa-base in financial domain for the first time.
Our model can be called Fin-RoBERTa.

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

Refer to the following documentation for basic code use.

[Basic code.md](https://github.com/deep-over/FiLM/blob/main/basic_code.md)


### Types of Training Corpora 📚
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

### Financial tasks performance
<table>
  <tr align='center'>
    <th class="tg-c3ow">Model</th>
    <th class="tg-c3ow" colspan="2">FPB</th>
    <th class="tg-c3ow">NER</th>
    <th class="tg-c3ow">Headline</th>
    <th class="tg-c3ow">FiNER</th>
    <th class="tg-c3ow" colspan="2">FinQA</th>
    <th class="tg-c3ow">FOMC</th>
  </tr>
  <tr align='center'>
    <td class="tg-c3ow">Metric</td>
    <td class="tg-c3ow">Accuracy</td>
    <td class="tg-c3ow">F-1</td>
    <td class="tg-c3ow">F-1</td>
    <td class="tg-c3ow">F-1</td>
    <td class="tg-c3ow">F-1</td>
    <td class="tg-c3ow">Prog Acc</td>
    <td class="tg-c3ow">Exe Acc</td>
    <td class="tg-c3ow">F-1</td>
  </tr>
  <tr align='center'>
    <td class="tg-c3ow">BERT [Devlin et al., 2019]</td>
    <td class="tg-c3ow">83.30</td>
    <td class="tg-c3ow">81.73</td>
    <td class="tg-c3ow">75.09</td>
    <td class="tg-c3ow">89.54</td>
    <td class="tg-c3ow">79.40</td>
    <td class="tg-c3ow">51.09</td>
    <td class="tg-c3ow">53.10</td>
    <td class="tg-c3ow">63.81</td>
  </tr>
  <tr align='center'>
    <td class="tg-c3ow">RoBERTa-base [Liu et al., 2019b]</td>
    <td class="tg-c3ow">85.30</td>
    <td class="tg-c3ow">83.93</td>
    <td class="tg-c3ow">78.81</td>
    <td class="tg-c3ow">91.29</td>
    <td class="tg-c3ow">81.58</td>
    <td class="tg-c3ow">56.76</td>
    <td class="tg-c3ow">59.11</td>
    <td class="tg-c3ow">69.16</td>
  </tr>
  <tr align='center'>
    <td class="tg-c3ow">Fin-BERT [Araci D et al., 2019]</td>
    <td class="tg-c3ow">85.25</td>
    <td class="tg-c3ow">82.45</td>
    <td class="tg-c3ow">77.93</td>
    <td class="tg-c3ow">90.48</td>
    <td class="tg-c3ow">81.49</td>
    <td class="tg-c3ow">47.86</td>
    <td class="tg-c3ow">50.04</td>
    <td class="tg-c3ow">64.50</td>
  </tr>
  <tr align='center'>
    <td class="tg-c3ow">Fin-BERT [Yang Y et al., 2020]</td>
    <td class="tg-c3ow">83.68</td>
    <td class="tg-c3ow">82.52</td>
    <td class="tg-c3ow">70.40</td>
    <td class="tg-c3ow">90.83</td>
    <td class="tg-c3ow">81.08</td>
    <td class="tg-c3ow">38.79</td>
    <td class="tg-c3ow">40.54</td>
    <td class="tg-c3ow">64.30</td>
  </tr>
  <tr align='center'>
    <td class="tg-c3ow">FLANG-BERT [Shah et al., 2022]</td>
    <td class="tg-c3ow">84.76</td>
    <td class="tg-c3ow">83.12</td>
    <td class="tg-c3ow">75.58</td>
    <td class="tg-c3ow">91.06</td>
    <td class="tg-c3ow">81.53</td>
    <td class="tg-c3ow">49.17</td>
    <td class="tg-c3ow">51.44</td>
    <td class="tg-c3ow">64.93</td>
  </tr>
  <tr align='center'>
    <td class="tg-c3ow">FLANG-RoBERTa [Shah et al., 2022]</td>
    <td class="tg-c3ow">83.86</td>
    <td class="tg-c3ow">82.18</td>
    <td class="tg-c3ow">71.36</td>
    <td class="tg-c3ow">90.46</td>
    <td class="tg-c3ow">80.78</td>
    <td class="tg-c3ow">30.69</td>
    <td class="tg-c3ow">32.17</td>
    <td class="tg-c3ow">68.02</td>
  </tr>
  <tr align='center'>
    <td class="tg-c3ow">SEC-BERT-base [Loukas L et al., 2022]</td>
    <td class="tg-c3ow">84.37</td>
    <td class="tg-c3ow">82.18</td>
    <td class="tg-c3ow">78.74</td>
    <td class="tg-c3ow">90.52</td>
    <td class="tg-c3ow">82.35</td>
    <td class="tg-c3ow">53.18</td>
    <td class="tg-c3ow">55.45</td>
    <td class="tg-c3ow">65.06</td>
  </tr>
  <tr align='center'>
    <td class="tg-c3ow">FiLM [ours]</td>
    <td class="tg-c3ow">86.25</td>
    <td class="tg-c3ow">84.48</td>
    <td class="tg-c3ow">79.78</td>
    <td class="tg-c3ow">91.79</td>
    <td class="tg-c3ow">82.02</td>
    <td class="tg-c3ow">58.85</td>
    <td class="tg-c3ow">61.38</td>
    <td class="tg-c3ow">69.60</td>
  </tr>
  <tr align='center'>
    <td class="tg-c3ow">FiLM (5.5B) [ours]</td>
    <td class="tg-c3ow">86.14</td>
    <td class="tg-c3ow">84.11</td>
    <td class="tg-c3ow">78.82</td>
    <td class="tg-c3ow">91.74</td>
    <td class="tg-c3ow">82.39</td>
    <td class="tg-c3ow">59.37</td>
    <td class="tg-c3ow">61.64</td>
    <td class="tg-c3ow">69.16</td>
  </tr>
</table>

### Information from financial tasks
<table class="tg">
<thead>
  <tr>
    <th class="tg-uzvj">Name</th>
    <th class="tg-uzvj">Task</th>
    <th class="tg-uzvj">Train size</th>
    <th class="tg-uzvj">Valid size</th>
    <th class="tg-uzvj">Test size</th>
    <th class="tg-uzvj">Metric</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td class="tg-uzvj">FPB [1]</td>
    <td class="tg-9wq8">Sentiment classification</td>
    <td class="tg-yz93">3,391 </td>
    <td class="tg-yz93">726 </td>
    <td class="tg-yz93">726</td>
    <td class="tg-9wq8">Accuracy &amp; F-1</td>
  </tr>
  <tr>
    <td class="tg-uzvj">NER [2]</td>
    <td class="tg-9wq8">Named entity recognition</td>
    <td class="tg-yz93">932</td>
    <td class="tg-yz93">232</td>
    <td class="tg-yz93">302</td>
    <td class="tg-9wq8">F-1</td>
  </tr>
  <tr>
    <td class="tg-uzvj">Headline [3]</td>
    <td class="tg-9wq8">News headlines classification</td>
    <td class="tg-yz93">7,989</td>
    <td class="tg-yz93">1,141</td>
    <td class="tg-yz93">2,282</td>
    <td class="tg-9wq8">F-1</td>
  </tr>
  <tr>
    <td class="tg-uzvj">FiNER [4]</td>
    <td class="tg-9wq8">Numeric entity recognition</td>
    <td class="tg-yz93">900,384</td>
    <td class="tg-yz93">112,494</td>
    <td class="tg-yz93">108,378</td>
    <td class="tg-9wq8">F-1</td>
  </tr>
  <tr>
    <td class="tg-uzvj">FinQA [5]</td>
    <td class="tg-9wq8">Question answering</td>
    <td class="tg-yz93">6,251</td>
    <td class="tg-yz93">883</td>
    <td class="tg-yz93">1,147</td>
    <td class="tg-9wq8">Accuracy(Prog &amp; Exe)</td>
  </tr>
  <tr>
    <td class="tg-uzvj">FOMC [6]</td>
    <td class="tg-9wq8">Sentiment classification</td>
    <td class="tg-yz93">1,588</td>
    <td class="tg-yz93">396</td>
    <td class="tg-yz93">496</td>
    <td class="tg-9wq8">F-1 (Combined-S)</td>
  </tr>
</tbody>
</table>

For information on the task, refer to the [FLUE benchmark](https://github.com/SALT-NLP/FLANG). We follow Benchmark too.

[1] https://huggingface.co/datasets/financial_phrasebank

[2] https://huggingface.co/datasets/tner/fin

[3] https://www.kaggle.com/datasets/ankurzing/sentiment-analysis-in-commodity-market-gold/data

[4] https://github.com/nlpaueb/finer

[5] https://github.com/czyssrs/FinQA

[6] https://github.com/gtfintechlab/fomc-hawkish-dovish
