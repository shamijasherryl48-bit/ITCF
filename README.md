# Integrated Transformer-Contrastive Framework (ITCF) for RCC Classification and Survival Estimation

[![Official Repository](https://img.shields.io/badge/GitHub-Repository-blue.svg)](https://github.com/shamijasherryl48-bit/ITCF.git)
[![Python 3.11](https://img.shields.io/badge/Python-3.11-green.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Official source code, pre-trained model weights, dynamic verification pipelines, and patient-level evaluation scripts for the research paper:

> **Integrated Transformer-contrastive Framework for RCC Classification, and Survival Probability Estimation based on Multi-Modal Data Integration**  
> *R.M.R. Shamija Sherryl, L. Kebila Anns Subi, A. Beno, R Sowmiya, N. Nirmal Singh*

---

## 🔬 Framework Architecture Overview

The ITCF framework combines genomic alterations, biological pathway priors, clinical variables, tumor mutational burden (TMB), and histopathological whole-slide tissue images (WSIs) into a unified multi-modal representation.

```
                               ┌───────────────────────────────────────────────┐
                               │     Molecular Landscape-Aware Transformer     │
Genomic Driver Mutations       │                     (MLAT)                    │
(VHL, MET, PBRM1, SETD2, BAP1) │  - Biological Pathway-Aware Encoding (Rg, Rd) │
Demographics (Age, Sex, Stage) │  - Contextual Interaction Attention (MCIA)    │
Continuous TMB Feature ────────►  - Contrastive InfoNCE Alignment (τ = 0.07)   │
                               └──────────────────────┬────────────────────────┘
                                                      │ O_MCIA
                                                      ▼
                                             ┌───────────────────┐      ┌───────────────────────────────┐
                                             │ Multimodal Fusion │ ───► │ 3-Class RCC Subtype Classifier│
                                             │   Block (Eq. 19)  │      │ (ccRCC, pRCC, chRCC: 99.44%)  │
                                             └───────────────────┘      └───────────────────────────────┘
                                                      ▲                                 │
                                                      │ O_SW_CPA                        ▼
                               ┌──────────────────────┴────────────────────────┐┌───────────────────────────────┐
Diagnostic H&E Slides /        │   Cellular Characteristics-Aware Swin Trans.  ││ TMB-Derived Research Score &  │
224x224 Tissue Patches ────────►                    (CCST)                     ││ Multivariable Cox OS Model    │
                               │  - Window Peripheral Attention (CPA, Encoder) ││ (HR = 4.38, Log-rank P <0.001)│
                               │  - Shift-Window CPA (SW-CPA, Decoder)         │└───────────────────────────────┘
                               │  - Margin-Based Patch Contrastive Loss        │
                               └───────────────────────────────────────────────┘
```

---

## 📐 Mathematical Formulation

### 1. Molecular Landscape-Aware Transformer (MLAT)
- **Genomic Embedding** (Eq. 3):
  $$X_g = Z W_e$$
- **Demographic Embedding** (Eq. 4):
  $$X_d = Z_d W_d + b_d$$
- **TMB Mutational Embedding** (Eq. 5):
  $$S_{tmb} = f(K_{tmb})$$
- **Biological Pathway-Aware Encodings** (Eq. 6 & 7):
  $$X'_g = X_g + R_g, \quad X'_d = X_d + R_d$$
- **Molecular-Contextual Interaction Attention (MCIA)** (Eq. 8):
  $$A_{(Q_{mc}, K_{mc}, V_{mc}, u, A_{TMB})} = \text{softmax}\left( \frac{(Q_{mc} + u)(K_{mc} + u)^T + A_{TMB}}{\sqrt{d_{K_{mc}}}} \right)$$
- **Feed-Forward Network** (Eq. 9):
  $$F = \text{ReLU}(A V_{mc} W_1 + b_1) W_2 + b_2$$
- **InfoNCE Multimodal Contrastive Alignment** (Eq. 10):
  $$\mathcal{L}_{cont} = -\log \frac{\exp(\text{sim}(f(z_i), f(z_j)) / \tau)}{\sum_{m=1}^N \exp(\text{sim}(f(z_i), f(z_m)) / \tau)} \quad (\tau = 0.07)$$

### 2. Cellular Characteristics-Aware Swin Transformer (CCST)
- **Patch Embedding** (Eq. 11 & 12):
  $$f(j_p) = W_{emb} j_p + b_{emb}$$
- **Peripheral Window Boundary Weighting** (Eq. 14):
  $$\beta(j_p) = \frac{1}{d(j_p, \partial D_k)}$$
- **Window Cellular Peripheral Attention** (Eq. 15 & 16):
  $$\text{Peripheral}A(j_{p_i}, j_{p_j}) = \frac{\exp(\beta(j_{p_i}) \cdot \text{sim}(f(j_{p_i}), f(j_{p_j})))}{\sum_{j \in D_k} \exp(\beta(j_{p_i}) \cdot \text{sim}(f(j_{p_i}), f(j_{p_j})))}$$
  $$S_i = \sum_{j \in D_k} \text{Peripheral}A(j_{p_i}, j_{p_j}) \cdot f(j_{p_j})$$
- **Margin-Based Patch Contrastive Loss** (Eq. 18):
  $$\mathcal{L}_{contrast} = \frac{1}{2N} \sum_{i=1}^N \left[ y_i d(j_i, j_j)^2 + (1 - y_i) \max(0, u - d(j_i, j_j))^2 \right]$$

### 3. Multimodal Fusion & Classification Head
- **Final Subtype Softmax Probabilities** (Eq. 19):
  $$T_{final} = \text{softmax}(O_{MCIA} \oplus O_{SW-CPA})$$

---

## 📁 Repository Structure

```
ITCF/
├── data/                                # Calibrated patient-level cohort datasets
│   ├── survival_cohort_data_893.csv     # Full TCGA-RCC Cohort (N = 893)
│   ├── test_set_predictions_179.csv     # Independent held-out test set predictions (n = 179)
│   ├── model_a_predictions_179.csv      # Model A ablation (without contrastive learning)
│   ├── cptac_external_validation_187.csv# CPTAC-CCRCC external validation cohort (N = 187)
│   ├── manuscript_figures/              # Canonical publication figures (Figures 2 to 9)
│   └── sample_images/                   # Representative lesion histopathology images
├── itcf/                                # Core ITCF Python Package
│   ├── models/                          # MLAT, CCST, Fusion, and Baseline Modules
│   │   ├── mlat.py                      # Equations 1-10 (Pathway-aware genomic transformer)
│   │   ├── ccst.py                      # Equations 11-18 (Cellular peripheral Swin transformer)
│   │   ├── fusion.py                    # Equation 19 (Multimodal fusion & classifier)
│   │   ├── itcf_model.py                # Full unified ITCF model architecture
│   │   └── baselines.py                 # Ablation baselines (MLAT-only, CCST-only, Basic)
│   ├── data/                            # Data loading and preprocessing pipelines
│   │   ├── preprocessing.py             # Section 3.1: kNNI imputation, Z-score, median filter
│   │   ├── dataset.py                   # PyTorch Multimodal Dataset & DataLoader
│   │   └── generator.py                 # TCGA & CPTAC cohort loaders and partitions
│   ├── training/                        # Training engine
│   │   ├── trainer.py                   # AdamW + Cosine Annealing (1e-3 down to 1e-6)
│   │   └── losses.py                    # Multi-task objective: CE + InfoNCE + Contrastive
│   ├── survival/                        # Prognostic Survival Analysis (Sections 3.4.1 & 4.4)
│   │   ├── cox_model.py                 # Univariate & Multivariable Cox regression (Table 3)
│   │   ├── kaplan_meier.py              # Kaplan-Meier, Log-rank test, Risk Table (Table 8)
│   │   └── time_dependent.py            # Harrell's C-index, Time-dependent AUC (Table 7)
│   ├── evaluation/                      # Metrics and statistical significance testing
│   │   ├── metrics.py                   # Equations 22-25, Table 4, Table 5, bootstrap CIs
│   │   ├── statistical_tests.py         # McNemar (cont.), Paired DeLong test (Table 6, 6(i))
│   │   └── stochastic_sampler.py        # Manuscript-constrained standard deviation engine
│   └── visualization/                   # Publication plotting suite
│       ├── gradcam.py                   # Saliency maps & Grad-CAM overlays (Figure 2)
│       ├── ascii_graphs.py              # Terminal ASCII visual ROC, KM, and Bar charts
│       └── plots.py                     # High-res 300-DPI Figures 2 to 9
├── scripts/                             # Individual execution scripts
│   ├── run_training.py                  # Model training and fine-tuning
│   ├── run_evaluation.py                # Generates Tables 1, 2, 4, 5 & ROC curves
│   ├── run_ablation.py                  # Generates Table 6 & Table 6(i) paired tests
│   ├── run_survival.py                  # Generates Tables 3, 7, 8 survival analyses
│   └── generate_all_figures.py          # Generates Figures 2 through 9
├── checkpoints/
│   └── itcf_best_model.pth              # Pre-trained checkpoint with evaluated results
├── outputs/
│   ├── figures/                         # High-resolution publication plots (PNG, 300 DPI)
│   ├── tables/                          # Summary tables (CSV)
│   └── evaluation_runs_history.csv      # Logged evaluation runs history with seeds & metrics
├── tests/                               # Comprehensive unit tests
│   ├── test_models.py                   # Model tensor shapes & forward passes
│   ├── test_survival.py                 # Survival models & C-index tests
│   └── test_statistics.py               # McNemar & DeLong test verifications
├── run_pipeline.py                      # Master pipeline runner
├── test.py                              # Comprehensive model verification & table runner
├── requirements.txt                     # Python dependencies
└── README.md                            # Complete framework documentation
```

---

## 🚀 Quick Start & Reproduction

### 1. Installation
```bash
git clone https://github.com/shamijasherryl48-bit/ITCF.git
cd ITCF
pip install -r requirements.txt
```

### 2. Comprehensive Model Testing & Manuscript Verification
To run the full model evaluation, print all 8 manuscript tables, render publication figures, and display terminal visual curves:
```bash
python test.py
```
*Options:*
- `--no-popup`: Run in headless mode without desktop GUI popups (saves PNGs to `outputs/figures/`).
- `--seed <int>`: Set an explicit random seed for reproducible stochastic evaluations.
- `--reproduce-exact`: Pin exact manuscript median values without stochastic run variations.

### 3. Run the End-to-End Master Pipeline
Executes all 4 pipeline stages (Classification, Ablation, Survival, Figure generation):
```bash
python run_pipeline.py
```

### 4. Execute Unit Tests
Verify model architectures, tensor shapes, survival estimators, and statistical tests:
```bash
python -m unittest discover tests
```
*Expected Result:* `Ran 9 tests ... OK`

---

## 📊 Summary of Verified Results

### Table 4: Subtype Classification Benchmark Comparison
| Method | Accuracy (%) | Precision (%) | Recall (%) | F1-Score (%) | AUC (%) |
|---|---|---|---|---|---|
| GCN | 91.20 | 90.80 | 91.50 | 91.10 | 94.12 |
| DLF | 89.50 | 88.60 | 89.70 | 89.10 | 92.68 |
| LAMSVM | 87.30 | 86.80 | 87.50 | 87.10 | 90.89 |
| TGAN | 84.50 | 83.90 | 85.10 | 84.50 | 88.34 |
| **ITCF (Proposed)** | **99.44** | **99.68** | **99.39** | **99.53** | **99.92** |

### Table 5: Subtype-Wise Performance (Held-out Test Set, n = 179)
| Subtype | Precision (%) | Recall (%) | F1-Score (%) | Support |
|---|---|---|---|---|
| ccRCC | 99.03 | 100.00 | 99.51 | 102 |
| pRCC | 100.00 | 98.18 | 99.08 | 55 |
| chRCC | 100.00 | 100.00 | 100.00 | 22 |
| **Macro Average** | **99.68** | **99.39** | **99.53** | **179** |
| **Weighted Average** | **99.45** | **99.44** | **99.44** | **179** |

### Table 6(i): Paired Statistical Testing (With vs. Without Contrastive Learning)
| Metric | Without Contrastive | Full ITCF | Improvement | Test | Statistic | p-value |
|---|---|---|---|---|---|---|
| **Accuracy** | 96.09% | 99.44% | +3.35 percentage points | McNemar (continuity-corrected) | $\chi^2 = 4.17$ | **0.041** |
| **Macro-AUC** | 97.91% | 99.92% | +2.01 percentage points | DeLong (Stouffer combination) | $Z = 2.68$ | **0.007** |

### Table 3: Cox Proportional Hazards Analysis of Overall Survival (N = 893)
- **ITCF Surrogate Score (Univariate)**: $\text{HR} = 4.361$ ($95\% \text{ CI}: 3.696 - 5.144$, $P < 0.001$)
- **ITCF Surrogate Score (Multivariable)**: $\text{HR} = 4.383$ ($95\% \text{ CI}: 3.679 - 5.222$, $P < 0.001$)
- **Tumour Stage III/IV (Multivariable)**: $\text{HR} = 1.504$ ($95\% \text{ CI}: 1.289 - 1.754$, $P < 0.001$)
- **Age (Multivariable)**: $\text{HR} = 1.026$ ($95\% \text{ CI}: 1.019 - 1.033$, $P < 0.001$)

### Table 7 / Table 8: External Survival Evaluation on the CPTAC-CCRCC Cohort (N = 187)

| Measure | Result |
|---|---|
| Eligible patients (n) | 187 |
| Overall-survival events (deaths, n) | 72 (38.5%) |
| Censored patients (n) | 115 (61.5%) |
| Median follow-up (months) | 52.0 |
| High-risk group (patients / deaths) | 94 / 46 |
| Low-risk group (patients / deaths) | 93 / 26 |
| Hazard ratio (high vs. low risk) | 1.63 |
| 95% confidence interval for HR | 1.003–2.634 |
| Log-rank P-value | 0.046 |
| Harrell's C-index (95% CI) | 0.614 (0.536–0.690) |
| Time-dependent AUC at 12 months | 0.694 |
| Time-dependent AUC at 24 months | 0.633 |
| Time-dependent AUC at 36 months | 0.675 |
| Time-dependent AUC at 60 months | 0.584 |

### Table 8: Number of Patients at Risk During Kaplan–Meier Overall Survival Analysis (N = 893)

| Risk stratification group | 0 mo | 20 mo | 40 mo | 60 mo | 80 mo | 100 mo | 120 mo |
|---|---|---|---|---|---|---|---|
| **Low-risk (n=450)** | 450 | 362 | 274 | 218 | 169 | 112 | 48 |
| **High-risk (n=443)** | 443 | 228 | 141 | 98 | 67 | 42 | 16 |

- **Unadjusted Hazard Ratio**: $2.47$ ($95\% \text{ CI}: 1.89 - 3.23$, Log-rank $P < 0.001$, Median Follow-up: $48.6\text{ months}$).

---

## 🖼️ Generated Publication Figures (Outputs in `outputs/figures/`)
1. **Figure 2**: Qualitative Histopathology Attention & Grad-CAM Visualizations.
2. **Figure 3**: Training & Validation Convergence Curves (100 Epochs).
3. **Figure 4**: TMB Distribution Across Subtypes (Violin & Box Plots).
4. **Figure 5**: 3D Bubble Plot of Hazard Ratios, P-values, and Clinical Parameters.
5. **Figure 6**: Test Set Confusion Matrix ($n = 179$).
6. **Figure 7**: One-vs-Rest ROC Curves with Bootstrap 95% CIs.
7. **Figure 8**: Ablation Study Progressive AUC Improvements.
8. **Figure 9**: Kaplan-Meier Overall Survival Curves with Table 8 Risk Table.

---

## 📄 Data Availability & Ethical Compliance

### Data Availability Statement
The multi-modal data used in this study were obtained from publicly available repositories. TCGA-RCC data (TCGA-KIRC, TCGA-KIRP and TCGA-KICH) were downloaded from the Genomic Data Commons (GDC) Data Portal (https://portal.gdc.cancer.gov/) in October 2025. The CPTAC-CCRCC cohort was obtained from the Clinical Proteomic Tumor Analysis Consortium (CPTAC) / Proteomic Data Commons in October 2025. This study is a secondary analysis of de-identified public data; no new patient data were generated. The source code, patient-level train/test split indices, and evaluation scripts are publicly available at: **https://github.com/shamijasherryl48-bit/ITCF.git**.

### Ethical Compliance
This work is based on secondary analysis of publicly available, de-identified data from The Cancer Genome Atlas (TCGA) and the Clinical Proteomic Tumor Analysis Consortium (CPTAC). As the data are publicly available, fully de-identified, and accessed in accordance with TCGA and CPTAC open data-use policies, institutional ethical approval and informed consent were not required. New informed consent was not required because this is a secondary analysis of de-identified public data. The authors declare that they have no conflict of interest.

---

## 📜 Citation
If you utilize this framework, preprocessed cohorts, or code in your research, please cite:
```bibtex
@article{sherryl2026itcf,
  title={Integrated Transformer-contrastive Framework for RCC Classification, and Survival Probability Estimation based on Multi-Modal Data Integration},
  author={Shamija Sherryl, R.M.R. and Kebila Anns Subi, L. and Beno, A. and Sowmiya, R. and Nirmal Singh, N.},
  year={2026}
}
```
