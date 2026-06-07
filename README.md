# Lightweight LLM Summarization with LSTM for Early ICU Deterioration Prediction

Reference implementation for the manuscript:

> **Lightweight LLM summarization with LSTM for early ICU deterioration prediction**
> Mohammadreza Momenzadeh, Atiyeh Oshaghi

This repository contains the code, SQL extraction logic, prompt templates, model
definitions, training/evaluation pipeline, and configuration needed to reproduce
the multimodal ICU deterioration prediction framework described in the paper.

The system combines **template-guided Flan-T5-Large summarization** of clinical
notes with a **multi-task bidirectional LSTM** over temporal vital signs and
laboratory values, fused to jointly predict in-hospital mortality,
vasopressor-dependent shock, and mechanical ventilation within 48 hours of a
24-hour prediction window.

---

## ⚠️ Data access

This project uses **MIMIC-IV v2.2** and the **MIMIC-IV-Note** module, which are
*credentialed-access* datasets distributed via PhysioNet. **No patient data is
included in this repository.** You must obtain access independently:

- MIMIC-IV: https://physionet.org/content/mimiciv/2.2/
- MIMIC-IV-Note: https://physionet.org/content/mimic-iv-note/2.2/

Access requires completion of the CITI "Data or Specimens Only Research" training
and signing the PhysioNet credentialed data use agreement.

---

## Repository structure

```
icu-llm-lstm/
├── README.md
├── LICENSE
├── CITATION.cff
├── requirements.txt
├── environment.yml
├── .gitignore
├── config/
│   └── default_config.yaml      # all hyperparameters (matches Supplementary S1)
├── sql/
│   ├── 01_cohort_identification.sql
│   ├── 02_vitalsigns_extraction.sql
│   └── 03_labs_extraction.sql
├── src/
│   ├── __init__.py
│   ├── data_extraction.py       # cohort build, exclusion flow
│   ├── preprocessing.py         # imputation, scaling, sequence assembly
│   ├── note_processing.py       # section extraction, cleaning, deid checks
│   ├── llm_summarizer.py        # Flan-T5-Large template-guided summarization
│   ├── embedding_pipeline.py    # 5-field -> 256-dim embedding
│   ├── models.py                # BiLSTM + attention + fusion + multi-task heads
│   ├── losses.py                # focal loss
│   ├── train.py                 # training loop
│   ├── evaluate.py              # AUROC/AUPRC/ECE/DCA/DeLong
│   ├── sensitivity.py           # observation-window / horizon / note-availability
│   └── utils.py                 # seeding, metrics helpers
├── scripts/
│   ├── run_full_pipeline.sh
│   └── reproduce_tables.sh
└── data/
    └── itemid_reference.csv     # itemid -> feature mapping (no patient data)
```

## Quickstart

```bash
# 1. Create environment
conda env create -f environment.yml
conda activate icu-llm-lstm

# 2. Configure database connection and paths
cp config/default_config.yaml config/local_config.yaml
# edit local_config.yaml: PostgreSQL credentials, output dirs

# 3. Build cohort and extract features (requires local MIMIC-IV PostgreSQL)
python -m src.data_extraction   --config config/local_config.yaml
python -m src.preprocessing     --config config/local_config.yaml

# 4. Summarize notes (offline, batched)
python -m src.note_processing   --config config/local_config.yaml
python -m src.llm_summarizer    --config config/local_config.yaml
python -m src.embedding_pipeline --config config/local_config.yaml

# 5. Train and evaluate
python -m src.train     --config config/local_config.yaml
python -m src.evaluate  --config config/local_config.yaml

# 6. Sensitivity analyses (reviewer-requested)
python -m src.sensitivity --config config/local_config.yaml
```

## Reproducibility

- All random seeds fixed (`seed: 42` in config); see `src/utils.set_seed`.
- Hyperparameters in `config/default_config.yaml` correspond one-to-one with
  Supplementary Section S1 of the manuscript.
- Hardware used in the paper: single NVIDIA RTX 3090 (24 GB), AMD Ryzen 9 5950X,
  64 GB RAM, Ubuntu 22.04, CUDA 11.8.

## Computational footprint (as reported)

| Stage | Mode | Cost |
|---|---|---|
| Note summarization (full cohort) | Offline (batch) | ~18 h |
| Multimodal LSTM training | Offline | 4–6 h, peak 19.2 GB VRAM |
| Single-patient inference | Real-time | 12–18 ms, 2.1 GB VRAM |

## License

Code released under the MIT License (see `LICENSE`). MIMIC-IV data is **not**
covered by this license and is governed by the PhysioNet credentialed DUA.


