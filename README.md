# Lipophilicity Prediction using MoLFormer-XL
This project explores transfer learning for predicting lipophilicity (logD) of chemical compounds using SMILES representations and a pre-trained chemical language model, **MoLFormer**. We investigate multiple fine-tuning techniques and data selection strategies to improve regression performance.

## Overview

The project is organized into three main tasks:

**Task 1 - Fine-Tuning the Pretrained Model:**
   - Load and fine-tune the MoLFormer model on the Lipophilicity dataset.
   - Apply unsupervised fine-tuning using Masked Language Modeling (MLM).
   - Evaluate performance using MSE, RMSE, MAE, and R² metrics.

**Task 2 - Influence Function-Based Data Selection:**
   - Compute influence scores for external dataset samples.
   - Select top-k samples based on influence and fine-tune the model with them.
   - Use LiSSA to approximate the inverse Hessian for efficient influence score computation.

**Task 3 - Diversity-Based Data Selection and Alternative Fine-Tuning:**
   - Apply clustering techniques (K-Means, Hierarchical) to select diverse samples.
   - Compare performance using BitFit, LoRA, and iA³ fine-tuning techniques.

## Repository Structure

```
LipophilicityPrediction/
│
├── Task1.ipynb # Fine-tuning and baseline regression
├── Task2.ipynb # Influence function-based data selection
├── Task3.ipynb # Clustering and fine-tuning strategy comparison
├── tasks/
│ └── External-Dataset_for_Task2.csv # External dataset used in Task 2
│ └── Task1.txt # Description of Task1
│ └── Task2.txt # Description of Task2
│ └── Task3.txt # Description of Task3
├── requirements.txt
└── README.md
```

## Datasets

- **Lipophilicity Dataset**: From MoleculeNet benchmark, includes SMILES and logD values.
- **External Dataset**: Contains additional SMILES and lipophilicity values for training enrichment.

## Libraries & Tools

- Python, PyTorch, Hugging Face Transformers, Scikit-learn, Weights & Biases (wandb)
- NumPy, Pandas, Matplotlib, Datasets, ipywidgets, Jupyter

## Performance Metrics (Sample)

| Method    | MSE    | RMSE   | MAE    | R² Score |
|-----------|--------|--------|--------|----------|
| Baseline  | 0.4157 | 0.6447 | 0.4881 | 0.7048   |
| FT Reg    | 0.4192 | 0.6475 | 0.4899 | 0.7023   |
| BitFit    | 1.3135 | 1.1461 | 0.9217 | 0.0671   |
| LoRA FT   | 0.4298 | 0.6556 | 0.5030 | 0.6947   |    <-
| iA3       | 0.4472 | 0.6687 | 0.5038 | 0.6824   |

## How to Run

1. Clone the repository:

```bash
git clone https://github.com/yourusername/LipophilicityPrediction.git
cd LipophilicityPrediction
```

2. Install dependencies:

```
pip install -r requirements.txt
```

3. Open notebooks and run step-by-step in Jupyter:

```
jupyter notebook
```

## Authors

- Mohammad Shaique Solanki
- Nikolai Egorov
- Chaitanya Jobanputra
