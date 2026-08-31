Breast Cancer Classification & Intelligent ML Agent 🔬
This project builds an end-to-end Machine Learning pipeline for breast cancer diagnosis using the **Breast Cancer Wisconsin Diagnostic dataset** (UCI Repository). 
It compares **SVM** and **KNN** models and evaluates how data preprocessing (Feature Selection, PCA) improves accuracy. The entire workflow is automated through a custom **Intelligent Agent**.

🛠️ Tech Stack
* **Language:** Python 3.x
* **Machine Learning:** `scikit-learn`
* **Data & Math:** `pandas`, `numpy`
* **Visualization:** `matplotlib`, `seaborn`
* **Dataset Loader:** `ucimlrepo`

🔄 Intelligent Agent Workflow (`IntelligentClassificationAgent`)
1. **Data Exploration:** Checks shape, missing values, and target balance.
2. **Preprocessing:** Removes duplicates, applies `StandardScaler`, selects top 20 features with `SelectKBest`, and reduces dimensions to 10 components using `PCA`.
3. **Training & Evaluation:** Evaluates models across 5 iterations.
4. **Automated Selection:** Picks the optimal model based on **F-measure**.
5. **Real-Time Inference:** Predicts diagnosis for new patient cases.
