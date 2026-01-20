#DBMIF: A Deep Balanced Multi-modal Iterative Fusion Framework for Air- and Bone-Conduction Speech Enhancement
This repository is the official project page for **DBMIF**, a framework for audio (AC) and bone-conduction (BC) based speech enhancement with iterative fusion and balanced interaction.

> **Code release plan:**  
> The full implementation (training/inference scripts, configuration files) will be released **upon paper acceptance** to ensure a clean, well-documented, and reproducible codebase.

---

## Paper

- **Title:** *[Your Paper Title Here]*
- **Venue:** *[Journal/Conference Name]* (under review / major revision)
- **Project page / PDF:** *TBD*

If you use this work, please cite our paper (citation will be updated after acceptance).

---

## Overview

DBMIF follows the end-to-end pipeline:

**PQMF Analysis → DIAF → CBGI (Encoder) → DBI → CBGI (Decoder) → PQMF Synthesis**

Key components:
- **PQMF subband processing** for efficient time-domain reconstruction
- **DIAF**: early iterative fusion for the fusion branch
- **CBGI**: cross-branch gated interaction (encoder-side and decoder-side)
- **DBI**: deep balanced interaction with equilibrium-style refinement

---

## Release Contents (Planned)

After acceptance, this repository will include:
- Full training and inference code
- Data preparation scripts
- Reproducibility configs and instructions
- (Optional) evaluation scripts and logs
