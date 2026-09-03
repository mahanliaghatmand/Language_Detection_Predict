# 🎬 Hollywood Movie Rating Predictor

A regression project that predicts a Hollywood movie's **IMDb-style rating (1–10)** from production and release attributes — budget, box-office gross, genre, director experience, runtime, release season, and sequel status — using a feed-forward neural network built with **TensorFlow / Keras**.

![Python](https://img.shields.io/badge/Python-3.12-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-Keras-orange)
![License](https://img.shields.io/badge/License-MIT-green)

---

## 📌 Overview

This project frames movie-rating prediction as a **supervised regression problem**: given a set of numeric features describing a film, the model learns to output a single continuous value — the predicted rating. It's a compact, end-to-end example of the regression workflow with Keras: data loading → model definition → training → evaluation → inference on a new sample.

> ⚠️ **Scope note:** the dataset has **64 rows** and was synthetically generated for learning purposes. This project is best understood as a hands-on exercise in building and evaluating a regression neural network, not as a production-ready movie-rating system. See [Limitations](#️-limitations--future-work) for details and concrete next steps.

---

## 🗂️ Project Structure

```
Hollywoodmovie_Rating_Model/
│
├── Data.csv                              # Dataset (64 movie records, 7 features + target)
├── Hollywoodmovie_Rating_Model.ipynb      # Data loading, model training & evaluation
├── LICENSE
└── README.md
```

---

## 📊 Dataset

- **Size:** 64 rows × 8 columns (7 features + 1 target)
- **Target:** `Rating` — a continuous score, roughly in the 3–8.6 range in this dataset
- **Format:** all columns are already numeric — categorical attributes (`genre`, `release_season`) are stored as **integer-encoded codes** rather than raw text

| Feature | Type | Description |
|---|---|---|
| `budget_million` | Numeric | Production budget, in million USD |
| `worldwide_gross_million` | Numeric | Worldwide box-office revenue, in million USD |
| `genre` | Numeric (encoded) | Integer code representing the movie's genre |
| `director_experience` | Numeric | Number of prior films / years of experience of the director |
| `runtime_min` | Numeric | Runtime in minutes |
| `release_season` | Numeric (encoded) | Integer code representing the release season |
| `sequel` | Binary | `1` if the movie is a sequel, `0` otherwise |
| `Rating` | Numeric (target) | The label the model is trained to predict |

---

## 🧠 Model Architecture

The notebook builds and trains **two feed-forward neural networks (Multi-Layer Perceptrons)** to compare a shallower and a slightly deeper architecture:

**`model`** — 2 layers
```
Dense(7, activation="relu", input_shape=[7])
Dense(1, activation="relu")   # output layer
```

**`model0`** — 3 layers
```
Dense(7, activation="relu", input_shape=[7])
Dense(7, activation="relu")
Dense(1, activation="relu")   # output layer
```

Both networks are:
- **Compiled** with the **Adam** optimizer and **Mean Squared Error (MSE)** loss (the standard loss for regression)
- **Trained** for **500 epochs** directly on the raw feature matrix `x` (no feature scaling is applied before training)

---

## 📈 Evaluation

The held-out test split (20% of the data, via `train_test_split`) is used to score `model0` with three standard regression metrics:

| Metric | Value | What it means |
|---|---|---|
| **MAE** (Mean Absolute Error) | **0.48** | On average, predicted ratings are off by ~0.48 points from the true rating |
| **RMSE** (Root Mean Squared Error) | **0.56** | Close to the MAE, suggesting errors are fairly consistent with no dramatic outliers |
| **R² Score** | **0.70** | The model explains ~70% of the variance in movie ratings — a solid result given only 64 training examples |

*(Since the split is random and not seeded, exact numbers will vary slightly between runs.)*

---

## 🚀 How to Run

1. Clone the repository and make sure `Data.csv` is in the working directory.
2. Open `Hollywoodmovie_Rating_Model.ipynb` (originally written for Google Colab — the file-upload cell can be swapped for a local `pd.read_csv("Data.csv")` when running elsewhere).
3. Run the cells in order:
   - Load the data and split it into features (`x`) and target (`y`)
   - Build, compile, and train the model(s)
   - Split into train/test and evaluate on the held-out set
   - Predict the rating for a new, hand-crafted movie sample

**Requirements:** `tensorflow`, `pandas`, `numpy`, `scikit-learn`

---

## ⚠️ Limitations & Future Work

This project is a learning exercise, and there are several known gaps worth being upfront about — both for transparency and as a roadmap for improvement:

- **No feature scaling:** inputs like `budget_million` (hundreds) and `director_experience` (single digits) are fed to the network on very different scales, which can slow down and destabilize training. Adding standardization (e.g. `StandardScaler`) is a natural next step.
- **Ordinal-encoded categoricals:** `genre` and `release_season` are plain integer codes, which implicitly tells the model one genre is "greater than" another. One-hot encoding would remove this false ordering.
- **ReLU activation on the output layer:** since ReLU clips negative values to zero, it can prevent the model from predicting low ratings correctly. A linear (no activation) output layer is more standard for unconstrained regression targets.
- **Single random train/test split:** with only 64 samples, one split can be misleadingly optimistic or pessimistic. K-fold cross-validation would give a more reliable performance estimate.
- **Synthetic, small dataset:** the data was generated for demo purposes rather than sourced from a real catalog (e.g. IMDb/Kaggle), so generalization to real movies is untested.
- **Two models, one used for evaluation:** the notebook trains both `model` and `model0`, evaluates `model0`, but the final inference cell predicts with `model`. Worth aligning which model is the "final" one.

**Ideas to extend this project:**
- Collect a larger, real-world dataset
- Add scaling + one-hot encoding to the preprocessing pipeline
- Compare against baseline models (Linear Regression, Random Forest, XGBoost)
- Add Dropout / early stopping to guard against overfitting
- Track experiments (e.g. with a fixed random seed) for reproducible comparisons

---

## 📄 License

Licensed under the **MIT License** — see [LICENSE](LICENSE) for details.

---

## 🙌 Acknowledgements

Dataset synthetically generated for educational and demonstration purposes.
