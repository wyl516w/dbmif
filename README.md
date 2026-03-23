# DBMIF: A Deep Balanced Multi-modal Iterative Fusion Framework for Air- and Bone-Conduction Speech Enhancement

## Paper

- **Title:** *DBMIF: a deep balanced multimodal iterative fusion framework for air- and bone-conduction speech enhancement*
- **Authors:** Yilei Wu, Changyan Zheng, Xingyu Zhang, Yakun Zhang, Chengshi Zheng, Shuang Yang, Ye Yan, Erwei Yin
- **Journal:** *Applied Intelligence*
- **Published:** March 18, 2026
- **DOI:** [10.1007/s10489-026-07150-z](https://link.springer.com/article/10.1007/s10489-026-07150-z)

## Key Points

- DBMIF targets speech enhancement in extremely low-SNR conditions by jointly using **air-conduction (AC)** and **bone-conduction (BC)** signals.
- The model is built as a **three-branch architecture** on top of a multi-scale interactive encoder-decoder backbone.
- It combines an **iterative attention module**, a **cross-branch gated module**, and a **balanced-interaction bottleneck** to enable adaptive AC-BC fusion and stable representation learning.
- Experiments show competitive gains in speech quality and intelligibility across multiple noise conditions, and the method reduces downstream **CER by at least 2.5%** against competing approaches.

## Training

Set dataset paths in `config/data_config/default_data.yaml`, then run:

```bash
bash train.sh
```

```powershell
.\train.ps1 --skip-gpu-check
```

```bat
train.bat --skip-gpu-check
```

Default model config: `config/model_config/DBMIF.yaml`

## Testing

Run:

```bash
bash test.sh DBMIF
```

```powershell
.\test.ps1 DBMIF
```

```bat
test.bat DBMIF
```

Outputs are written to `./results`.

## Citation

```bibtex
@article{wu2026dbmif,
  title   = {DBMIF: a deep balanced multimodal iterative fusion framework for air- and bone-conduction speech enhancement},
  author  = {Yilei Wu and Changyan Zheng and Xingyu Zhang and Yakun Zhang and Chengshi Zheng and Shuang Yang and Ye Yan and Erwei Yin},
  journal = {Applied Intelligence},
  year    = {2026},
  doi     = {10.1007/s10489-026-07150-z}
}
```
