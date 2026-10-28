# Boston Housing — Regression, PCA & Logistic Classification

**Reconstructed academic assignment:** based on the exact coursework instructions preserved in a 2025 ChatGPT conversation. **The original `BostonHousing.csv` file was not included in the export or current recoverable attachments**, so repository results are not claimed. The code is runnable with the original CSV if you have it.

## Original assignment (recovered from student's message)

**Regression target `MEDV`:** A) remove `TAX` and fit Linear Regression; B) select nonredundant variables by correlation and refit; C) PCA without standardization retaining over 95% variance; D) PCA with standardization retaining over 95% variance. Compare A–D.

**Classification target `EXPENSIVE`:** derive `EXPENSIVE = 1` when `MEDV >= 30` and `0` otherwise. Exclude `TAX` and `MEDV` from predictors. Compare Logistic Regression under (i) raw inputs, (ii) MinMax normalization and (iii) Z-score standardization.

## Implementation

`analysis.py` implements **all seven requested variants** using identical held-out splits for meaningful comparison. Correlation selection, imputation, scaling and PCA are all fitted **only on training data**. Results saved to `results/` when run.

```bash
pip install -r requirements.txt
pytest -q
python analysis.py BostonHousing.csv --output-dir results
```

The original conversation contained a prior *assistant-reported* Regression A result (MSE 25.40 and R² 0.654). These values have **not** been independently verified and are not used in this repository.

## Data availability and ethics

No original dataset is bundled. The tests generate deterministic, synthetic data **only to test the code paths**. Running this on the real coursework dataset is still needed before publishing authentic quantitative findings.
