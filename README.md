# Rating-Plausibility-of-Word-Senses-in-Ambiguous-Sentences
This project tries to solve SemEval-2026 Task 5: Rating Plausibility of Word Senses in Ambiguous Sentences through Narrative Understanding, where the task is to predict how plausible a candidate word is within short narratives rather than exact sense classification.

# Narrative-Level Word Sense Plausibility Prediction

## CS445 Natural Language Processing - Group 4 Final Project

This repository contains our final group project for **CS445 Natural Language Processing**. The project focuses on **SemEval-2026 Task 5: Rating Plausibility of Word Senses in Ambiguous Sentences through Narrative Understanding**.

The goal of this project is to predict how plausible a candidate word sense is within a short narrative. Unlike traditional Word Sense Disambiguation tasks, where the system chooses one correct sense for an ambiguous word, this task requires predicting a continuous plausibility score. Each candidate meaning is rated based on how well it fits the full story context.

---

## Project Description

Natural language systems often struggle with ambiguous words because word meanings can change depending on the surrounding story, implied events, and broader narrative context. Humans usually do not interpret ambiguous words in a strictly binary way. In many cases, multiple meanings can be possible, but some are more plausible than others.

This project approaches the problem as a **regression task**. Given a five-sentence narrative and a candidate sense gloss, the model predicts a plausibility score between **1 and 5**, where higher values indicate that the candidate meaning is more plausible in the story.

We implemented and compared two gloss-informed transformer-based architectures:

1. **Cross-Encoder Model**
2. **Bi-Encoder Model**

We also tested an ensemble approach that combines both models.

---

## Task

The task is based on narrative-level word sense plausibility prediction.

Each input example includes:

- A short narrative
- A target ambiguous word
- A candidate word sense gloss
- An example sentence for the candidate sense
- Human plausibility ratings

The model must estimate how plausible the candidate sense is in the given narrative.

---

## Dataset

We used the official dataset provided for **SemEval-2026 Task 5**.

Each sample contains the following fields:

- `precontext`
- `sentence`
- `ending`
- `homonym`
- `judged_meaning`
- `example_sentence`
- `average`
- `stdev`

The narrative was constructed by combining:

```text
precontext + sentence + ending
```

The candidate gloss was enriched by combining the gloss definition with the example sentence:

```text
judged_meaning + " For example: " + example_sentence
```

This enrichment gives the model more semantic information about the candidate sense.

### Dataset Statistics

| Split | Number of Entries |
|---|---:|
| Training Set | 2,280 |
| Development Set | 588 |
| Test Set | 930 |

The dataset contains **220 unique homonyms**.

Since the test set gold labels are not publicly available, all evaluation results in this project are reported on the development set.

---

## Methodology

### 1. Preprocessing

The first step was to prepare the story and gloss inputs for the models.

The three narrative fields were merged into one complete story:

```text
story = precontext + sentence + ending
```

The gloss was enriched with the example sentence:

```text
enriched_gloss = judged_meaning + " For example: " + example_sentence
```

Numeric fields such as `average` and `stdev` were converted into valid numeric values. Rows with missing labels were removed before training.

Tokenization was performed using the Hugging Face `bert-base-uncased` tokenizer. The maximum sequence length was set to 256 tokens.

---

## Model 1: Cross-Encoder

The first model is a **GlossBERT-style Cross-Encoder**.

In this architecture, the complete story and the enriched gloss are given to BERT together as a sentence pair:

```text
[CLS] story [SEP] enriched gloss [SEP]
```

This allows the transformer attention mechanism to directly compare the narrative context with the candidate sense definition.

The model uses:

- `bert-base-uncased`
- Hugging Face `AutoModelForSequenceClassification`
- Regression output layer
- `num_labels = 1`
- `problem_type = regression`

The `[CLS]` token representation is passed to a regression head that predicts the plausibility score.

### Cross-Encoder Training Settings

| Parameter | Value |
|---|---|
| Backbone Model | BERT-base |
| Learning Rate | 2e-5 |
| Weight Decay | 0.01 |
| Epochs | 8 |
| Warmup Steps | 72 |
| Loss Function | Mean Squared Error |
| Main Metric | Spearman Correlation |

The best checkpoint was selected according to development set Spearman correlation.

---

## Model 2: Bi-Encoder

The second model is a **Gloss-Informed Bi-Encoder** inspired by previous work on scalable WSD systems.

Unlike the cross-encoder, the bi-encoder encodes the story and gloss separately.

The architecture works as follows:

1. Encode the story using BERT.
2. Encode the enriched gloss using BERT.
3. Extract the `[CLS]` embedding from both.
4. Project both embeddings into a shared latent space.
5. Compute cosine similarity.
6. Concatenate the projected embeddings and similarity score.
7. Pass the result into a regression head.

The goal of this model is to compare the story and gloss representations in a shared embedding space.

### Bi-Encoder Training Settings

| Parameter | Value |
|---|---|
| Backbone Model | BERT-base |
| Learning Rate | 2e-5 |
| Weight Decay | 0.01 |
| Epochs | 8 |
| Loss Function | Mean Squared Error |
| Optimizer | AdamW |

Although this architecture is more efficient and modular, it performed worse than the cross-encoder in our experiments.

---

## Ensemble Model

We also tested a weighted ensemble of the cross-encoder and bi-encoder predictions.

The ensemble formula was:

```text
final_prediction = 0.7 * cross_encoder_prediction + 0.3 * bi_encoder_prediction
```

The cross-encoder was given a higher weight because it had stronger individual performance.

The ensemble slightly improved Accuracy@Std, but it did not significantly improve Spearman correlation.

---

## Evaluation Metrics

The models were evaluated using the following metrics:

### Spearman Correlation

Spearman correlation measures how well the model ranks candidate senses compared to human judgments. This was the main evaluation metric.

### Accuracy@Std

Accuracy@Std checks whether the model prediction falls within the acceptable range based on annotator agreement.

### Mean Squared Error

MSE measures the average squared difference between predicted and gold plausibility scores.

### Macro F1, Precision, and Recall

For additional analysis, continuous predictions were rounded into integer bins from 1 to 5 and evaluated as classification outputs.

---

## Results

### Development Set Results

| Model | Spearman | Accuracy@Std | MSE |
|---|---:|---:|---:|
| Cross-Encoder BERT Baseline | 0.3706 | 0.5510 | 1.4210 |
| Bi-Encoder BERT Baseline | 0.1364 | 0.4847 | 1.7293 |
| Cross-Encoder BERT + Example Sentence | 0.4374 | 0.5680 | 1.3506 |
| Bi-Encoder BERT + Example Sentence | 0.1073 | 0.5170 | 1.4921 |
| Ensemble 0.7 CE + 0.3 BE | 0.4385 | 0.5799 | - |

The best individual model was the **Cross-Encoder with example sentence enrichment**.

The best overall Spearman score was achieved by the ensemble model, but the improvement over the cross-encoder was very small.

---

## Best Model

Our best individual model:

```text
Model: Cross-Encoder BERT + Example Sentence
Spearman: 0.4374
Accuracy@Std: 0.5680
MSE: 1.3506
```

This result shows that direct interaction between the story and candidate gloss is very important for narrative-level plausibility prediction.

---

## RoBERTa Experiments

We also tested `roberta-base` as an alternative backbone model for the cross-encoder.

| Configuration | Spearman | Accuracy@Std | MSE |
|---|---:|---:|---:|
| RoBERTa, lr=1e-5, 8 epochs | 0.3775 | 0.5170 | 1.6695 |
| RoBERTa, lr=2e-5, 5 epochs | 0.3107 | 0.5272 | 1.5035 |

RoBERTa did not improve the results. This suggests that the main limitation was not model capacity, but the small dataset size.

---

## Key Findings

The cross-encoder consistently outperformed the bi-encoder.

The most important improvement was adding the `example_sentence` field to the gloss representation. This increased the cross-encoder Spearman score from **0.3706** to **0.4374**.

The bi-encoder struggled because it encoded the story and gloss separately. This created an information bottleneck and caused the model to produce narrow mid-range predictions.

The ensemble model improved Accuracy@Std slightly, but the bi-encoder predictions were too weak to provide a meaningful Spearman improvement.

RoBERTa did not outperform BERT, which suggests that more data would likely be more helpful than simply using a larger or different pretrained model.

---

## Error Analysis

The model struggled most with cases where lexical overlap misled the prediction. For example, if a story contains words related to a certain topic, the model may incorrectly assign high plausibility to a sense that is lexically related but semantically wrong.

Some of the hardest homonyms for the cross-encoder were:

| Homonym | Cross-Encoder MAE |
|---|---:|
| coached | 1.794 |
| docked | 1.627 |
| tipped | 1.563 |
| rare | 1.536 |
| reservations | 1.491 |
| try | 1.487 |
| pick | 1.427 |
| blaze | 1.380 |
| tips | 1.336 |
| dribbled | 1.329 |

These examples show that the task requires deeper narrative reasoning beyond surface-level word matching.

---

## Limitations

This project has several limitations:

- The training dataset is small.
- The test set gold labels are not publicly available.
- The bi-encoder collapses toward mid-range predictions.
- MSE does not fully capture the ordinal structure of the 1–5 rating scale.
- The models may rely on lexical overlap instead of true semantic reasoning.
- Larger models did not improve results due to limited training data.

---

## Future Work

Possible future improvements include:

- Using pairwise ranking loss instead of simple regression
- Applying contrastive learning for the bi-encoder
- Training with a larger dataset
- Using ordinal regression for the 1–5 plausibility scale
- Trying better ensemble strategies
- Adding more enriched gloss examples
- Using larger language models with more data
- Performing more detailed error analysis by homonym type
- Improving the model’s ability to handle narrative-level reasoning

---

## Technologies Used

- Python
- PyTorch
- Hugging Face Transformers
- BERT
- RoBERTa
- pandas
- NumPy
- scikit-learn
- Matplotlib

---

## Project Structure

```text
.
├── data/
│   ├── train.csv
│   ├── dev.csv
│   └── test.csv
│
├── models/
│   ├── cross_encoder.py
│   └── bi_encoder.py
│
├── notebooks/
│   └── experiments.ipynb
│
├── results/
│   ├── predictions.csv
│   └── evaluation_results.csv
│
├── README.md
└── requirements.txt
```

Note: The exact file names may differ depending on the local implementation.

---

## How to Run

### 1. Clone the Repository

```bash
git clone <repository-url>
cd <repository-name>
```

### 2. Create a Virtual Environment

```bash
python -m venv venv
```

Activate the environment:

For macOS/Linux:

```bash
source venv/bin/activate
```

For Windows:

```bash
venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

If `requirements.txt` is not available, install the main dependencies manually:

```bash
pip install torch transformers pandas numpy scikit-learn matplotlib scipy
```

### 4. Run Training

To train the cross-encoder:

```bash
python models/cross_encoder.py
```

To train the bi-encoder:

```bash
python models/bi_encoder.py
```

### 5. Evaluate the Models

Evaluation is performed on the development set using:

- Spearman correlation
- Accuracy@Std
- MSE
- Macro F1
- Macro Precision
- Macro Recall

---

## Group Members

- Azizcan Altınkan
- Bahar Akbaş
- Candan Tuna Mesut
- Tuana Doğan
- Umay Dzhummanov

---

## Individual Contributions

- **Candan Tuna Mesut:** Cross-encoder architecture and implementation, report writing, and improvement of evaluation score.
- **Tuana Doğan:** Bi-encoder architecture and implementation, including `BiEncoderModel` and `BiEncoderDataset` classes, and report writing.
- **Umay Dzhummanov:** Literature review, related work, citation alignment, and presentation preparation.
- **Bahar Akbaş:** Gloss enrichment strategy using example sentences and report writing.
- **Azizcan Altınkan:** Data loading, data preprocessing, and dataset statistics.

---

## Conclusion

In this project, we implemented and compared cross-encoder and bi-encoder transformer architectures for narrative-level word sense plausibility prediction.

The cross-encoder achieved the strongest individual performance because it directly models the interaction between the story and candidate gloss. Adding example sentences to the gloss representation was the most effective improvement.

The bi-encoder was more efficient but less accurate, mainly because it encoded the story and gloss separately. The ensemble model slightly improved Accuracy@Std but did not provide a meaningful Spearman improvement.

Overall, the project shows that direct context-gloss interaction is important for ambiguous word sense plausibility prediction, especially when the task requires understanding the full narrative rather than only a single sentence.

---

## References

1. Janosch Gehring et al. SemEval-2026 Task 5: Rating Plausibility of Word Senses in Ambiguous Sentences through Narrative Understanding.
2. Terra Blevins and Luke Zettlemoyer. Moving Down the Long Tail of Word Sense Disambiguation with Gloss Informed Bi-encoders.
3. Luyao Huang, Chi Sun, Xipeng Qiu, and Xuanjing Huang. GlossBERT: BERT for Word Sense Disambiguation with Gloss Knowledge.
4. Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding.
5. Boon Peng Yap, Andrew Koh, and Eng Siong Chng. Adapting BERT for Word Sense Disambiguation with Gloss Selection Objective and Example Sentences.
