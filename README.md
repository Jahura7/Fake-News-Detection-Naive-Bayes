# Fake News Detection Using Naive Bayes Classification

## Project Overview

This project focuses on developing a **Machine Learning classification model to distinguish between REAL and FAKE news articles** using the Multinomial Naive Bayes algorithm.

The project was completed as part of an assignment for the **Ostad Data Science and Machine Learning with Python Course**.

Using Natural Language Processing (NLP) techniques and TF-IDF feature extraction, the model learns patterns from news article text and predicts whether an article is REAL or FAKE.

## Objectives

- Explore and understand the news classification dataset.
- Perform data cleaning and preprocessing.
- Handle missing values and invalid records.
- Extract meaningful textual features using TF-IDF.
- Build a Multinomial Naive Bayes classification model.
- Evaluate model performance using accuracy, precision, recall, F1-score, and a confusion matrix.

## Dataset Description

**Dataset:** `fake_or_real_news.csv`

**Source:** Ostad Data Science and Machine Learning with Python Course – Assignment Dataset.

The dataset contains news article text and corresponding labels.

| Column | Description |
|---|---|
| `text` | News article content |
| `label` | Classification label (FAKE or REAL) |

The original dataset contains 7,795 records. After selecting the required columns and performing data cleaning, 6,279 valid observations were retained.

### Class Distribution

| Class | Number of Articles |
|---|---:|
| REAL | 3,161 |
| FAKE | 3,118 |
| **Total** | **6,279** |

The cleaned dataset contains a relatively balanced distribution of REAL and FAKE news articles.
 
## Development Environment

This project was developed and executed in **Google Colab**, a cloud-based Python notebook environment.

**Tools and Technologies:**
- Google Colab – Development and execution environment
- Python – Programming language
- Pandas and NumPy – Data manipulation
- Scikit-learn – Machine Learning and model evaluation
- Matplotlib – Data visualization
- TF-IDF – Text feature extraction
- Multinomial Naive Bayes – Classification algorithm


## Project Workflow

### 1. Data Loading and Exploration
- Imported the dataset using Pandas.
- Examined dataset structure, dimensions, data types, and missing values.
- Selected the required `text` and `label` columns.

### 2. Data Cleaning and Preprocessing
- Handled missing text values.
- Removed records with missing labels.
- Standardized label values using uppercase conversion.
- Retained only valid classes (REAL and FAKE).
- Converted article text into string format.
- Removed empty text records.

### 3. Feature Engineering

Used **TF-IDF (Term Frequency-Inverse Document Frequency)** to convert textual data into numerical features.

The following configurations were applied:

- English stop-word removal.
- `max_df=0.95`
- `min_df=2`
- `ngram_range=(1, 2)` to capture both individual words and two-word combinations.

### 4. Train-Test Split

The cleaned dataset was divided into:

- Training data: 80% (5,023 records)
- Testing data: 20% (1,256 records)

Used `random_state=42` for reproducibility and `stratify=y` to preserve class proportions.

### 5. Model Development

Developed a Machine Learning pipeline using:

**Multinomial Naive Bayes**

The pipeline combines:
- TF-IDF Vectorization
- Multinomial Naive Bayes Classifier

### 6. Model Evaluation

Evaluated the model using:

- Accuracy Score
- Precision
- Recall
- F1-Score
- Confusion Matrix

## Model Performance

The Multinomial Naive Bayes model achieved the following results on the test dataset:

<summary>
</summary>

| Evaluation Metric | Score |
|---|---:|
| Accuracy | **87.26%** |
| Macro Precision | 89% |
| Macro Recall | 87% |
| Macro F1-Score | 87% |

### Classification Report

| Class | Precision | Recall | F1-Score |
|---|---:|---:|---:|
| FAKE | 0.98 | 0.76 | 0.86 |
| REAL | 0.80 | 0.99 | 0.89 |

### Confusion Matrix

| Actual / Predicted | FAKE | REAL |
|---|---:|---:|
| FAKE | 473 | 151 |
| REAL | 9 | 623 |

### Key Findings

- The model achieved an overall test accuracy of 87.26%.
- REAL news articles were identified particularly well, with 99% recall.
- FAKE news predictions achieved 98% precision.
- However, FAKE news recall was 76%, indicating that some fake articles were incorrectly classified as REAL.

These findings highlight the importance of evaluating individual classification metrics rather than relying on accuracy alone.
## Repository Structure

```text
Fake-News-Detection-Naive-Bayes/
│
├── Fake-News-Detection-Naive-Bayes.ipynb
├── fake_or_real_news_dataset.zip
└── README.md
```

## How to Run the Project

This project can be accessed and executed using Google Colab.

**Step 1: Clone or download this repository**

```bash
git clone <your-github-repository-url>
```

**Step 2: Open Google Colab**

Visit: https://colab.research.google.com/

**Step 3: Upload the notebook**

Open `Naive_Bayes_Classification_Model_Assignment.ipynb` in Google Colab.

**Step 4: Upload the dataset**

Upload `fake_or_real_news.csv` to the Colab session or mount Google Drive and update the dataset file path accordingly.

**Step 5: Execute the notebook**

Run the notebook cells sequentially to perform data preprocessing, TF-IDF feature extraction, model training, and evaluation.

The required Python libraries can be installed directly in Google Colab if they are not already available.

 
## Conclusion

This project demonstrates the application of Natural Language Processing and supervised Machine Learning for text classification.

By combining TF-IDF feature extraction with a Multinomial Naive Bayes classifier, the model successfully classified news articles into REAL and FAKE categories with 87.26% test accuracy.

Although the model performed well overall, its lower recall for FAKE news indicates opportunities for further improvement.

This project provided practical experience in data exploration, data cleaning, feature engineering, Machine Learning pipeline development, and model evaluation.

## Learning Outcomes

Through this project, I gained practical experience in:

- Working with real-world text datasets.
- Data cleaning and preprocessing using Pandas.
- Text feature extraction using TF-IDF.
- Implementing Multinomial Naive Bayes.
- Building Scikit-learn pipelines.
- Evaluating classification models.
- Interpreting confusion matrices and classification reports.

---

**Project Type:** Machine Learning | Natural Language Processing | Text Classification

**Course:** Ostad – Data Science and Machine Learning with Python

**Author:** Jahura Begum
