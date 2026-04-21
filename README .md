# 📧 Email Spam Detection

![Python](https://img.shields.io/badge/Python-3.10-blue?style=flat-square&logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-1.35-red?style=flat-square&logo=streamlit)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.4-orange?style=flat-square&logo=scikit-learn)
![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)

A machine learning web app that classifies emails and SMS messages as **Spam** or **Ham (Legitimate)** using Multinomial Naive Bayes and TF-IDF vectorization — with a clean Streamlit GUI, deployable to Heroku.

---

## 🖥️ Live Demo

> Deploy to Heroku (see [Deployment](#-deployment) section below) and paste your Heroku URL here.

---

## 📁 Project Structure

```
email-spam-detection/
│
├── app.py                        # Streamlit web application
├── Email_Spam_Detection.ipynb    # Full notebook (EDA + training)
├── spam.csv                      # Dataset
├── model.pkl                     # Trained Naive Bayes model (generated)
├── vectorizer.pkl                # TF-IDF vectorizer (generated)
│
├── requirements.txt              # Python dependencies
├── Procfile                      # Heroku process file
├── setup.sh                      # Streamlit config for Heroku
└── .gitignore
```

---

## 🔍 How It Works

1. **Data Cleaning** — Drops unused columns, removes duplicates, renames labels
2. **EDA** — Pie chart, bar charts, distribution plots, pairplot, and word clouds
3. **Text Preprocessing** — HTML removal → special character removal → lowercase → stopword removal → stemming
4. **Balancing** — Undersampling to equalize ham/spam class counts
5. **Vectorization** — TF-IDF with 1500 features and bigrams `(1,2)`
6. **Model** — Multinomial Naive Bayes (`alpha=0.5`)
7. **Evaluation** — Accuracy score, classification report, confusion matrix heatmap

---

## 🚀 Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/email-spam-detection.git
cd email-spam-detection
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Train the model

Open and run all cells in `Email_Spam_Detection.ipynb`. This generates `model.pkl` and `vectorizer.pkl`.

```bash
jupyter notebook Email_Spam_Detection.ipynb
```

### 4. Run the Streamlit app

```bash
streamlit run app.py
```

The app opens at `http://localhost:8501`.

---

## ☁️ Deployment

### Deploy to Heroku

**Prerequisites:** [Heroku CLI](https://devcenter.heroku.com/articles/heroku-cli) installed and logged in.

```bash
# 1. Initialize git (if not already)
git init
git add .
git commit -m "Initial commit"

# 2. Create Heroku app
heroku create your-app-name

# 3. Push and deploy
git push heroku main

# 4. Open your live app
heroku open
```

> ⚠️ Make sure `model.pkl` and `vectorizer.pkl` are committed before pushing.  
> Remove them from `.gitignore` if needed:
> ```bash
> git add model.pkl vectorizer.pkl
> git commit -m "Add trained model artifacts"
> ```

---

## 📊 Dataset

- **Source:** [UCI SMS Spam Collection](https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset)
- **Size:** 5,572 messages (4,825 ham + 747 spam)
- **Columns used:** `v1` (label) and `v2` (message text)
- **After balancing:** 653 ham + 653 spam (undersampling)

---

## 🛠️ Tech Stack

| Component | Library |
|-----------|---------|
| Data manipulation | `pandas`, `numpy` |
| NLP / Text cleaning | `nltk` |
| Visualization | `matplotlib`, `seaborn`, `wordcloud` |
| ML model | `scikit-learn` (MultinomialNB) |
| Web app | `streamlit` |
| Deployment | Heroku |

---

## 📈 Model Performance

| Metric | Score |
|--------|-------|
| Accuracy | ~97% |
| Precision (spam) | ~96% |
| Recall (spam) | ~98% |

*(Results may vary slightly due to random sampling)*

---

## 🤝 Contributing

Pull requests are welcome! For major changes, please open an issue first to discuss what you'd like to change.

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).

---

<p align="center">Made with ❤️ using Python & Streamlit</p>
