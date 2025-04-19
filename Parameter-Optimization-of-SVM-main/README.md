
# SVM Optimization Experiment

## Overview
This repository contains an implementation of a Support Vector Machine (SVM) optimization experiment for multi-class classification. The experiment evaluates SVM performance across multiple samples with varying hyperparameters to identify the best-performing configuration.

## Features
- Synthetic multi-class dataset with 5000 samples and 10 features  
- SVM hyperparameter optimization for linear, RBF, and polynomial kernels  
- Evaluation across 10 different train/test splits  
- Visualizations for convergence, class distribution, and feature correlations  
- Export of results and graphs for GitHub showcasing  

## Dataset
- **5000 samples**
- **10 features** (7 informative, 3 redundant)
- **5 classes** (equally distributed)
- Standardization via `StandardScaler`

## Methodology
1. Generate a synthetic dataset using `make_classification`
2. Perform preprocessing (scaling and class balance checks)
3. Split the data into 10 different train/test sets (70/30)
4. Optimize SVM with:
   - **Linear kernel**: Varying C values  
   - **RBF kernel**: Varying C and gamma  
   - **Polynomial kernel**: Varying C and degree  
5. Track convergence history (accuracy vs iterations)
6. Store and visualize results

## Results

| Sample | Best Accuracy | Best SVM Parameters     |
|--------|---------------|--------------------------|
| S1     | 0.982         | linear, 0.464, 215.443   |
| S2     | 0.971         | rbf, 2.154, 0.005        |
| S3     | 0.994         | linear, 0.1, 0.005       |
| S4     | 0.988         | linear, 2.154, 0.005     |
| S5     | 0.994         | rbf, 2.154, 0.005        |
| S6     | 0.988         | linear, 0.022, 215.443   |
| S7     | 0.982         | rbf, 215.443, 0.001      |
| S8     | 0.982         | linear, 0.464, 2.154     |
| S9     | 0.988         | poly, 0.022, 0.464       |
| S10    | 0.994         | linear, 2.154, 0.001     |

**Best Accuracy**: 0.994 (achieved in S3, S5, S10)

## Visualizations
- `best_svm_convergence.png`: Accuracy vs iterations for best model  
- `class_distribution.png`: Class label distribution  
- `feature_correlation.png`: Correlation heatmap of features  

## Output Files
- `svm_optimization_results.csv`: Accuracy and parameters for all 10 samples  
- Plots saved as `.png` files for GitHub documentation  

## Dependencies
- Python 3.x  
- `numpy`  
- `pandas`  
- `scikit-learn`  
- `matplotlib`  
- `seaborn`  

## Usage

```bash
git clone https://github.com/yourusername/svm-optimization-experiment.git
cd svm-optimization-experiment
pip install -r requirements.txt
python svm_experiment.py
```

Or run in Google Colab:

👉 [Open in Colab](https://colab.research.google.com/drive/1W-NaCXbcTH51ech3jUUNiXSSjpb_K3aZ#scrollTo=6E1PAdI0kVvV)

## License
[MIT License](LICENSE)

## Contributing
Feel free to fork, contribute, and create pull requests.
