# language_detection_predict

### Overview
**language_detection_predict** is a machine learning project that detects the language of a given text among three languages: **English, Persian (Farsi), and Arabic**. The model is built using a **Gradient Boosting Classifier** from `scikit-learn`, with text features extracted via **TF-IDF**.

### Features
- Detects **3 languages**: English 🇬🇧, Persian 🇮🇷, Arabic 🇸🇦
- Text vectorization using **TF-IDF**
- Classification using **Gradient Boosting Classifier (sklearn)**
- Custom-collected dataset

### Tech Stack
- Python
- scikit-learn (`GradientBoostingClassifier`)
- TF-IDF Vectorizer

### Dataset
The dataset was **self-collected** (custom-built) covering samples of English, Persian, and Arabic text.

### Model Performance
| Metric | Score |
|--------|-------|
| Train Accuracy | 1.0 |
| Test Accuracy | 0.5 |

### How to Run
This project is provided as a **Jupyter Notebook**.

1. Clone the repository:
   ```bash
   git clone https://github.com/mahanly89mly-boop/Language_Detection_Predict.git
   cd language_detection_predict
   ```
2. Install dependencies:
   ```bash
   pip install scikit-learn pandas numpy jupyter
   ```
3. Open and run the notebook:
   ```bash
   jupyter notebook
   ```

### License
This project is licensed under the MIT License

---
