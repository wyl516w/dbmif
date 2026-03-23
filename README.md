# DBMIF: A Deep Balanced Multi-modal Iterative Fusion Framework for Air- and Bone-Conduction Speech Enhancement

This repository contains the training, inference, and evaluation code for **DBMIF**, a multi-modal speech enhancement framework that combines **air-conduction (AC)** and **bone-conduction (BC)** signals through iterative fusion.

---

## Paper

- **Title:** *DBMIF: a deep balanced multimodal iterative fusion framework for air- and bone-conduction speech enhancement*
- **Authors:** Yilei Wu, Changyan Zheng, Xingyu Zhang, Yakun Zhang, Chengshi Zheng, Shuang Yang, Ye Yan, Erwei Yin
- **Status:** Accepted by *Applied Intelligence*
- **Preprint:** [arXiv:2603.02877](https://arxiv.org/abs/2603.02877)
- **arXiv DOI:** [10.48550/arXiv.2603.02877](https://doi.org/10.48550/arXiv.2603.02877)
- **Submitted to arXiv:** March 3, 2026

The public journal page with final volume / issue / page information does not appear to be online yet, so the arXiv record is currently the most complete public metadata source.

---

## Overview

DBMIF follows an end-to-end enhancement pipeline built around sub-band analysis and cross-modal interaction:

**PQMF Analysis -> Iterative Fusion -> Cross-branch Interaction -> Balanced Refinement -> PQMF Synthesis**

The repository currently exposes the following main model-side entry names:

- `ACBC`: the main dual-branch AC/BC enhancement model
- `OnlyAC`: AC-only variant
- `OnlyBC`: BC-only variant
- `GaGNet`, `Conformer`, `DCCRN`, `DenGCAN`, `FCN`, `MMINet`: baseline / comparison models

Available model config files are under `config/model_config`:

- `DBMIF.yaml`
- `EBEN.yaml`
- `GaGNet.yaml`
- `Conformer.yaml`
- `DCCRN.yaml`
- `DenGCAN.yaml`
- `FCN.yaml`
- `MMINet.yaml`

---

## Repository Layout

```text
dbmif/
|- config/                 # data / trainer / model YAML configs
|- module/                 # datamodules, models, metrics, utilities
|- scripts/
|  |- test/                # inference, metrics, ASR, CER helpers
|  |- test.sh              # Linux unified test pipeline
|  `- test.ps1             # Windows unified test pipeline
|- dataset/                # dataset placeholders / documentation
|- main.py                 # training entry
|- train.sh                # Linux training entry
|- train.ps1               # Windows training entry
|- train.bat               # Windows cmd wrapper
|- test.sh                 # Linux test wrapper
|- test.ps1                # Windows test wrapper
`- test.bat                # Windows cmd wrapper
```

---

## Data Layout

The default configuration expects:

- AC/BC paired data under `./dataset/abdata/ABCS_database`
- noise data under `./dataset/noisedata`

The default datamodule configuration is in `config/data_config/default_data.yaml`. It uses:

- paired `ab` data
- `noise` data
- an `ab+noise` fusion datamodule
- train SNR sampled from `[-15, 5]`
- test SNR list `[-15, -5, 5]`

If your local dataset paths differ, update the YAML files before training or testing.

---

## Entry Points

Training:

- Linux: `train.sh`
- Windows PowerShell: `train.ps1`
- Windows CMD: `train.bat`

Testing:

- Linux: `test.sh`
- Windows PowerShell: `test.ps1`
- Windows CMD: `test.bat`

The default training config is `config/model_config/DBMIF.yaml`, and the default test output directory is `./results`.

---

## Evaluation

The unified test pipeline covers:

1. inference
2. reference generation
3. objective metrics
4. ASR transcription
5. CER calculation

Supporting scripts are located in `scripts/test`.

---

## Optional Dependencies

Core training / inference relies on PyTorch, PyTorch Lightning, Torchaudio, TorchMetrics, SoundFile, and YAML-related dependencies.

The ASR stage additionally expects:

- `wenet`
- `openai-whisper`

If these are not installed, the ASR step in the test pipeline will fail even if enhancement inference itself is available.

---

## Notes

- Windows `.ps1` and `.bat` entry scripts are provided alongside the Linux `.sh` scripts.
- Debug artifacts such as `tmp`, `results`, logs, checkpoints, and cache files are excluded through `.gitignore`.
- The repository currently tracks code and lightweight dataset placeholders only; full datasets should be prepared locally.

---

## Citation

```bibtex
@misc{wu2026dbmif,
  title         = {DBMIF: a deep balanced multimodal iterative fusion framework for air- and bone-conduction speech enhancement},
  author        = {Yilei Wu and Changyan Zheng and Xingyu Zhang and Yakun Zhang and Chengshi Zheng and Shuang Yang and Ye Yan and Erwei Yin},
  year          = {2026},
  eprint        = {2603.02877},
  archivePrefix = {arXiv},
  primaryClass  = {eess.AS},
  doi           = {10.48550/arXiv.2603.02877}
}
```
