# DBMIF: A Deep Balanced Multi-modal Iterative Fusion Framework for Air- and Bone-Conduction Speech Enhancement

This repository contains the training, inference, and evaluation code for **DBMIF**, a multi-modal speech enhancement framework that combines **air-conduction (AC)** and **bone-conduction (BC)** signals through iterative fusion.

The project currently includes:

- training entry scripts for Linux and Windows
- unified testing, inference, metric, and ASR evaluation scripts
- model configuration files for DBMIF and several baselines
- data module definitions for AC, BC, and noise fusion
- core model implementations for `ACBC`, `OnlyAC`, `OnlyBC`, and comparison models

---

## Paper

- **Title:** *DBMIF: a deep balanced multimodal iterative fusion framework for air- and bone-conduction speech enhancement*
- **Venue:** *Applied Intelligence* (Accept)
- **Project page / PDF:** *TBD*

If you use this repository, please cite the paper. The citation block can be updated once the final publication metadata is available.

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

## Training

### Linux

```bash
bash train.sh
```

### Windows PowerShell

```powershell
.\train.ps1 --dry-run --skip-gpu-check
.\train.ps1 --skip-gpu-check
```

### Windows CMD

```bat
train.bat --dry-run --skip-gpu-check
train.bat --skip-gpu-check
```

The default training entry is `main.py`, which merges:

- data config
- trainer config
- model config
- optional extra overrides from command-line arguments

The default model config is `config/model_config/DBMIF.yaml`.

---

## Testing and Evaluation

The unified test pipeline performs:

1. inference
2. reference text generation
3. objective metric computation
4. ASR transcription
5. CER calculation

### Linux

```bash
bash test.sh DBMIF
```

### Windows PowerShell

```powershell
.\test.ps1 DBMIF --dry-run
.\test.ps1 DBMIF
```

### Windows CMD

```bat
test.bat DBMIF --dry-run
test.bat DBMIF
```

The default output directory is `./results`.

If you want to test a different model or provide custom files:

```powershell
.\test.ps1 GaGNet --config .\config\model_config\GaGNet.yaml --ckpt path\to\model.ckpt
```

Testing helper scripts live under `scripts/test`.

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
@article{dbmif2026,
  title   = {DBMIF: a deep balanced multimodal iterative fusion framework for air- and bone-conduction speech enhancement},
  author  = {To be updated},
  journal = {Applied Intelligence},
  year    = {2026}
}
```
