# Data-Driven Road Safety Severity Prediction Using Ensemble Stacking 🚦🤖

## 📌 Project Overview

This project aims to predict the **severity of road traffic accidents** (Minor, Serious, or Fatal) using **machine learning** techniques. It leverages ensemble learning methods to improve prediction accuracy and support emergency response systems in real-time.

By combining multiple classifiers (MLP, Voting, and Stacking), this system captures complex patterns in crash-related data such as weather, road type, vehicle characteristics, and more.

---

## 🎯 Objectives

- Predict accident severity using numerical crash attributes.
- Implement and compare different ensemble machine learning models.
- Improve accuracy, robustness, and generalization using stacking.
- Deploy the model via a user-friendly web interface.

---

## 🧠 Algorithms Used

- **Multi-Layer Perceptron (MLP)** – Deep learning neural network model.
- **Voting Classifier** – Aggregates results from multiple models (hard/soft voting).
- **Ensemble Stacking** – Combines outputs of multiple base models via a meta-classifier.

---

## 🛠️ Technologies & Libraries

| Category        | Tools Used                               |
|----------------|-------------------------------------------|
| Language        | Python 3.x                                |
| Libraries       | Scikit-learn, Pandas, NumPy, Matplotlib, Seaborn |
| Development     | Jupyter Notebook, Google Colab, VS Code   |
| Web Interface   | Flask                                     |
| Model Saving    | Pickle / Joblib                           |
| Deployment      | Localhost / Conda Environment             |

---

## 🏗️ Project Structure

```

📁 road-safety-severity-prediction
├── data/
│   └── accidents.csv
├── models/
│   ├── mlp\_model.pkl
│   ├── voting\_model.pkl
│   └── stacking\_model.pkl
├── notebooks/
│   └── model\_training.ipynb
├── app/
│   ├── templates/
│   │   └── index.html
│   ├── static/
│   └── app.py
├── README.md
└── requirements.txt

````

---

## 🔁 Workflow

1. **Data Preprocessing**
   - Handle missing values, encode categorical features, normalize numerical data.

2. **Model Training**
   - Train individual classifiers and stack them into an ensemble.
   - Use k-fold cross-validation for evaluation.

3. **Evaluation Metrics**
   - Accuracy, Precision, Recall, F1-Score, Matthews Correlation Coefficient (MCC).

4. **Model Deployment**
   - Flask web app for interactive prediction.
   - Predicts severity based on user input or uploaded CSV.

---

## 📈 Results

| Model              | Accuracy |
|-------------------|----------|
| Gradient Boosting  | 81%      |
| Random Forest      | 78%      |
| **Ensemble Stacking** | **85%**  |

- Fewer false positives in fatal cases.
- Balanced performance across all severity classes.

---

## 🚀 Run Locally

```bash
# Clone the repository
git clone https://github.com/yourusername/road-safety-severity-prediction.git

# Navigate into the directory
cd road-safety-severity-prediction

# Install dependencies
pip install -r requirements.txt

# Run the Flask app
cd app
python app.py
````

---

## 🌐 Web Interface

* **Home Page** – Introduces the project
* **Prediction Page** – Input road crash parameters manually
* **Output Page** – View the predicted severity class (Minor, Serious, Fatal)

---

## 🧪 Testing

* Functional Testing (data input/output)
* Model Evaluation via Confusion Matrix
* Usability & Performance Tests
* Cross-browser frontend checks

---

## 🔮 Future Enhancements

* Integrate real-time GPS and sensor data.
* Use Explainable AI (XAI) to visualize feature impact.
* Expand model to support multilingual inputs.
* Deploy on cloud platforms for scalable access.

---

## 📄 License

This project is intended for educational and academic use only. Contact authors for extended use or collaboration.





