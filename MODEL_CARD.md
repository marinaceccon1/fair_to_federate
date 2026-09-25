# Fair to Federate? Two-Sided Prediction of Performance and Fairness Outcomes in Healthcare Federated Learning

> **Research use only. Not a medical device.**
> This software was developed in the HEREDITARY project as a research prototype. It has not been assessed or certified under the EU Medical Device Regulation (EU) 2017/745. It must not be used to diagnose, treat, monitor or make any other decision about individual patients.

## 1. Overview

| Field | Value |
|---|---|
| Name | `fair_to_federate` (code for the paper "Fair to Federate?") |
| Version | v2.0 (CIKM '26 camera-ready) |
| Type | Software or library (research code; no trained model weights and no data are distributed) |
| Owner partner and contact | University of Padova — Marina Ceccon, marina.ceccon@phd.unipd.it |
| License | MIT (see LICENSE file) |
| Repository and DOI | https://github.com/marinaceccon1/fair_to_federate |
| Release date | August 2026 |
| Related task or deliverable | N/A |
| How to cite | M. Ceccon, A. Fabris, O. Irrera, G. Silvello, G. A. Susto. "Fair to Federate? Two-Sided Prediction of Performance and Fairness Outcomes in Healthcare Federated Learning." Proc. CIKM '26, Rome, Italy, 2026. doi:10.1145/3799682.3840981 |

## 2. Intended use

**Primary intended use.** Research code that implements a two-stage meta-learning framework for the *Federation Assessment Problem*: estimating, from institution-level summary statistics only (dataset size, demographic composition, label-bias rate), how joining a federation would change predictive performance and group-fairness metrics, for both a candidate institution and the institutions already in the federation. It is a methodological study tool for reproducing the experiments of the associated paper and for extending the analysis to other datasets.

**Intended users.** Machine-learning and biomedical-informatics researchers, data scientists studying fairness and federated learning.

**Out-of-scope uses.** This release must not be used for
- clinical diagnosis, prognosis or treatment of individual patients;
- decisions about individuals, such as insurance, employment or access to care;
- re-identification of individuals or linkage with other data sources;
- deciding in practice whether a real hospital should join a real federation, without independent validation: the predictors reported in the paper are fitted on simulated clients drawn from the NIH and Fitzpatrick17k datasets;
- populations, devices or data types not described in Section 3.

## 3. Data provenance

| Dataset | Source partner | Type | Population | Ethics approval or access conditions |
|---|---|---|---|---|
| NIH ChestX-ray14 | External, public (NIH Clinical Center) | Public, de-identified | ~112k chest radiographs from ~30k patients, US, mixed sex and age, 14 thoracic pathologies; sex used as sensitive attribute | Public release by NIH; used under its own terms. Not redistributed here; users download it themselves |
| Fitzpatrick17k | External, public (Groh et al., 2021) | Public | ~16.5k clinical skin images labelled with Fitzpatrick skin type; skin tone used as sensitive attribute | Public dataset; used under its own terms. Not redistributed here; users download it themselves |

No data collected inside HEREDITARY and no partner data were used. Both datasets are public, already de-identified and obtained from their original public sources; no ethics approval was required for their secondary use in this methodological study.

This repository and the released files contain no personal data, pseudonymized records, identifiers or pseudonymization keys. No image data and no trained model weights are distributed: the code downloads nothing and expects the user to obtain the public datasets independently.

## 4. Method

- **Algorithm or architecture.** Two stages. Stage 1: controlled simulation — DenseNet-121 image classifiers trained standalone and federated (FedAvg, Flower, 30 rounds, full participation, 1 local epoch) over configurations that vary client size, demographic composition and injected label-bias rate. Stage 2: tabular meta-regressors (Linear Regression, Ridge, Random Forest, Gradient Boosting, XGBoost) that map candidate and federation meta-features to the metric-change vectors, 5-fold cross-validation with randomized hyperparameter search.
- **Training setting.** Federated (simulated, single-machine/multi-GPU) and centralized standalone baselines.
- **Privacy techniques.** Not applicable to the release itself; the method by design uses only aggregate institution-level metadata (size, composition, label-bias rate) and never raw data or model weights. No differential privacy or secure aggregation is implemented.

## 5. Performance

- **Evaluation data.** Held-out 15% test split of each source dataset; meta-regressors evaluated with 5-fold cross-validation over simulated federation configurations (729 two-client configurations; 800 sampled configurations each for three- and five-client federations, per domain). Additional out-of-distribution check on 150 seven-client configurations not seen in training.
- **Metrics and results.** Coefficient of determination R² on the metric-change targets. Candidate perspective, NIH: 0.961 (ΔAUC), 0.941 (ΔTPRgap); Fitzpatrick17k: 0.855 (ΔAUC), 0.944 (ΔTPRgap). Federation-members perspective, NIH: 0.957 (ΔAUC), 0.789 (ΔTPRgap); Fitzpatrick17k: 0.890 and 0.915. Seven-client extrapolation (NIH): 0.674 (ΔAUC), 0.649 (ΔBalAcc), 0.901 (ΔTPRgap), 0.865 (ΔminTPR). Linear models perform markedly worse, near zero for fairness targets.
- **Results by subgroup.** Fairness targets are themselves subgroup-based: TPR gap and minimum TPR across sex (NIH) and skin-tone (Fitzpatrick17k) groups. R² broken down by federation size is reported in the paper and in the supplementary material.

## 6. Known limitations and risks

- **Generalizability.** All clients in a federation are disjoint samples from the same source dataset. This isolates the effect of demographic composition and label bias but omits real cross-institutional heterogeneity (acquisition equipment, imaging protocols, case mix). Reported accuracy holds under controlled bias conditions and is not a guarantee in real deployments. Only two imaging domains were evaluated; the framework does not model client dropout.
- **Known biases and confounders.** Label bias is injected synthetically at a known rate; in practice it must be estimated and estimation error propagates into the predictions. Evaluation is limited to a single binary sensitive attribute per dataset (sex; light/dark skin tone), so intersectional subgroups are not covered. Both source datasets carry their own documented label noise and demographic imbalance. Federation-level meta-features are simple aggregates and can coincide for federations made of very different clients.
- **Failure modes.** Largest prediction errors occur for rare edge configurations, in particular NIH federations dominated by female samples combined with high label noise. Fairness metrics on Fitzpatrick17k required a soft-TPR variant because dark-skin positives are scarce in the test set; hard-TPR estimates are unstable there. Predictions outside the parameter grids and federation sizes used for fitting are extrapolations.
- **Privacy risks.** None identified for the released artefact: no personal data, no trained weights and no data-derived parameters are distributed, so membership-inference or model-inversion tests are not applicable. The method itself exchanges only aggregate counts and rates between parties; whether such aggregates could leak information in a real deployment was not tested and would need assessment before operational use.

## 7. Responsible use and reporting

- **Terms of use.** Source code only, released under the MIT license (see LICENSE). No trained model weights and no datasets are distributed, so no access control is needed. Use is restricted to research; the out-of-scope uses in Section 2 apply.
- **Contact for misuse or vulnerability reports.** marina.ceccon@phd.unipd.it

## 8. Acknowledgement

Funded by the European Union under grant agreement No 101137074 (HEREDITARY). Views and opinions expressed are however those of the author(s) only and do not necessarily reflect those of the European Union or the European Health and Digital Executive Agency (HaDEA). Neither the European Union nor the granting authority can be held responsible for them.
