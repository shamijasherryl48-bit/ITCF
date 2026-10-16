"""
Master Execution Pipeline for ITCF Framework
=============================================
Runs the complete end-to-end ITCF reproduction and verification workflow:
  1. Multi-modal Classification Evaluation (Tables 1, 2, 4, 5)
  2. Ablation Study & Paired Statistical Significance Testing (Table 6, Table 6(i))
  3. Prognostic Cox Proportional Hazards & Kaplan-Meier Analysis (Tables 3, 7, 8)
  4. Generation of all 300-DPI Publication Figures (Figures 2 to 9)
"""

import os
import sys
import subprocess
import time

if sys.stdout.encoding.lower() != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

os.environ['PYTHONIOENCODING'] = 'utf-8'


def print_banner(title: str):
    print("\n" + "=" * 90)
    print(f" {title.upper()}")
    print("=" * 90)


def main():
    start_time = time.time()
    print_banner("INTEGRATED TRANSFORMER-CONTRASTIVE FRAMEWORK (ITCF) PIPELINE")
    print("Paper: 'Integrated Transformer-contrastive Framework for RCC Classification,")
    print("        and Survival Probability Estimation based on Multi-Modal Data Integration'")
    print("Authors: R.M.R. Shamija Sherryl, L. Kebila Anns Subi, A. Beno, R Sowmiya, N. Nirmal Singh")
    
    base_dir = os.path.dirname(os.path.abspath(__file__))
    scripts_dir = os.path.join(base_dir, 'scripts')
    
    # Step 1: Run Classification Evaluation
    print_banner("Step 1: Multi-Modal Classification Evaluation (Tables 1, 2, 4, 5)")
    cmd_eval = [sys.executable, os.path.join(scripts_dir, 'run_evaluation.py')]
    subprocess.run(cmd_eval, check=True)
    
    # Step 2: Run Ablation & Paired Statistical Tests
    print_banner("Step 2: Ablation Study & Paired Tests (Table 6, Table 6(i))")
    cmd_ablation = [sys.executable, os.path.join(scripts_dir, 'run_ablation.py')]
    subprocess.run(cmd_ablation, check=True)
    
    # Step 3: Run Survival Analysis
    print_banner("Step 3: Prognostic Survival Analysis (Tables 3, 7, 8)")
    cmd_survival = [sys.executable, os.path.join(scripts_dir, 'run_survival.py')]
    subprocess.run(cmd_survival, check=True)
    
    # Step 4: Generate All Figures
    print_banner("Step 4: Generating Publication Figures (Figures 2 to 9)")
    cmd_figures = [sys.executable, os.path.join(scripts_dir, 'generate_all_figures.py')]
    subprocess.run(cmd_figures, check=True)
    
    elapsed = time.time() - start_time
    print_banner(f"PIPELINE COMPLETED SUCCESSFULLY IN {elapsed:.2f} SECONDS")
    print("All generated tables saved to: outputs/tables/")
    print("All generated figures saved to: outputs/figures/")
    print("=" * 90)


if __name__ == '__main__':
    main()
