🔍 Bug Predictor using Machine Learning

Bug-Predictor is an intelligent tool that leverages machine learning to identify bug-prone files in software repositories. By mining Git history and extracting static code metrics, it trains predictive models to surface high-risk areas in your codebase — helping developers prioritize testing, code review, and refactoring efforts.

🚀 Key Features

🧠 Machine Learning Models: Uses Random Forest and XGBoost for accurate bug prediction.

📊 Static Code Analysis: Extracts code complexity, churn, and other static metrics.

🕵️ Git History Mining: Analyzes commit patterns, file changes, and bug-introducing commits (BICs).

🔎 Risky File Identification: Highlights files most likely to contain bugs in future commits.

📈 Feature Importance: Offers insights into which metrics contribute most to predictions.

⚙️ Customizable Pipeline: Easily extendable with other ML models or metrics.

🎯 Use Cases

Software Quality Assurance: Focus testing efforts on the riskiest parts of your codebase.

Developer Productivity: Reduce time spent reviewing low-risk code.

Continuous Integration: Integrate predictions into your CI pipeline to flag high-risk changes.

📁 Input & Output

Input: A Git repository (local or remote).

Output:

Bug-prone files with risk scores

Visualizations of commit behavior

Feature importance graphs

🛠️ Technologies Used

Python

Scikit-learn

XGBoost

GitPython

Pandas, NumPy, Matplotlib

(Optional) DVC for data and model versioning
