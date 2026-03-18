# Project Summary (Submission Version)

## Project
**Attrition Risk Prediction Dashboard with Compensation Leverage Analysis**

## 1) Problem Statement
I built this dashboard to detect employee attrition risk early and identify where HR interventions—especially compensation actions—can be most effective.

## 2) Key Questions
- Which employees are likely to leave within ~12 months?
- Which departments/levels show concentrated risk?
- Where should compensation budget be prioritized (high absolute pay but low relative market position)?

## 3) Method
- Model: Logistic Regression (interpretability-first)
- Validation: Hold-out + repeated random split + Stratified K-fold
- Explainability: Odds Ratio (OR), SHAP, individual feature contribution
- Visualization: individual/team risk, pay competitiveness, 2D risk quadrant

## 4) Implementation Highlights
- Korean ERP column auto-mapping
- External industry wage matching by year + tenure group
- Combined use of absolute pay (salary_std) and relative pay (log ratios)
- Residualization to reduce multicollinearity between salary_std and external relative pay
- Team-level risk aggregation for active employees
- What-if simulator for fast policy scenario testing

## 5) Deliverables
1. Individual risk prediction panel
2. Department/position risk distribution
3. External/internal pay competitiveness analysis
4. Absolute-vs-relative pay risk quadrant
5. Deep-dive tabs (OR/SHAP/validation)

## 6) Business Use
- Prioritize retention targets
- Support compensation/promotion/job-rotation decisions
- Enable monthly/quarterly HR risk monitoring

## 7) Ownership & Build Process
- Problem framing, feature logic, interpretation framework, and HR action mapping were analyst-owned
- Coding was partially AI-assisted for speed
- Final interpretation and decision recommendations were reviewed and owned by me

## 8) Limitations
- Single-model baseline (logistic) may miss nonlinear effects
- Potential sample/time bias; external validation is needed
- Predictions should be used as decision support, not as standalone HR decisions
