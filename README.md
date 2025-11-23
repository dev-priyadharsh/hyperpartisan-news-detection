# Hyperpartisan News Detection (CW1 & CW2, 7120CEM)

## Overview

This project presents a comprehensive approach to hyperpartisan news detection using both classical machine learning (CW1: SVM, Random Forest, Logistic Regression) and modern deep learning (CW2: BERT). The work is based on the SemEval-2019 Task 4 "by-article" dataset, which includes labeled English news articles with marked hyperpartisan bias.

CW1: Classical ML pipeline using feature engineering and TF-IDF.

CW2: End-to-end deep learning with BERT fine-tuning for context-rich representation.

Both pipelines are fully reproducible in their respective Colab notebooks (see hyperpartisan_detection_CW1.ipynb and NLP_DL.ipynb).

## Highlights

- Dataset: SemEval-2019 Task 4 (source)
- Evaluation: Accuracy, F1-score, ROC-AUC, per-class recall/precision; visualizations included.
- Colab Notebooks: Annotated for learning and reproducibility.
- Results:
  - SVM baseline: ~71% accuracy, recall for minority (hyperpartisan) < 0.45.
  - BERT: 78.3% accuracy, macro-F1 0.76, recall for hyperpartisan 0.60.
 
## Getting Started
Requirements
- Google Colab (recommended)
- Python 3.8+
- Libraries: transformers, torch, sklearn, matplotlib, seaborn, lxml, tqdm

Running the Notebooks
- Open each notebook in Google Colab.
- Upload the dataset XML files.
- Follow the code blocks in order; markdown annotations will guide you.

## Results and Discussion

- CW1 Models: Showed strong performance on majority class but struggled to recall hyperpartisan content, highlighting limitations of TF-IDF and surface-level features.
- BERT (CW2): Achieved a significant boost in both overall accuracy and the crucial minority-class recall, due to its contextual embedding and transfer learning.
  Error analysis reveals BERT recognizes complex linguistic cues and subtle bias more reliably than classical methods.
