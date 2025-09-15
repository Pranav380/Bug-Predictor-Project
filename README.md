# Bug Predictor – Starter Project

Predict bug-prone files/functions by combining static code metrics, Git mining, and machine learning.
This starter gives you code for feature extraction, training, and a Streamlit dashboard.

## 0) Prerequisites
- Python 3.10+
- Git installed

## 1) Setup
```bash
# clone your target repo locally (example)
git clone https://github.com/your-org/your-repo.git ~/code/your-repo

# create venv
python -m venv .venv
# Windows: .venv\Scripts\activate
source .venv/bin/activate

# install deps
pip install -r requirements.txt
```

## 2) Build a dataset from a repo
```bash
python src/build_dataset.py --repo ~/code/your-repo --since_months 12 --out data/your_repo_dataset.csv
```

## 3) Train models
```bash
python src/model_train.py --data data/your_repo_dataset.csv --model_out models/bug_predictor.joblib
```

## 4) Run the dashboard
```bash
streamlit run app/streamlit_app.py -- --repo ~/code/your-repo --model models/bug_predictor.joblib --since_months 6
```

## 5) Project structure
```
.
├── app/
│   └── streamlit_app.py
├── data/
├── models/
├── notebooks/
├── src/
│   ├── build_dataset.py
│   ├── model_train.py
│   ├── miners/
│   │   └── git_miner.py
│   └── utils/
│       ├── static_metrics.py
│       └── io_utils.py
├── requirements.txt
└── README.md
```
