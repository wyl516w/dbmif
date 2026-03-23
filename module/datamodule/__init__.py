from .ab_datamodule import AB_DataModule, AB_Dataset 
from .abn_fusion_datamodule import FusionDataModule as AB_N_Fusion_datamodule, FusionDataset as AB_N_Fusion_Dataset
from .noise_datamodule import NoiseDataModule, NoiseDataset 

DATAMODULE_MAP = {
    "ab": AB_DataModule,
    "AB_DataModule": AB_DataModule,
    "noise": NoiseDataModule,
    "NoiseDataModule": NoiseDataModule,
    "ab+noise": AB_N_Fusion_datamodule,
    "AB_N_Fusion_datamodule": AB_N_Fusion_datamodule,
}

DATASET_MAP = {
    "ab": AB_Dataset,
    "AB_Dataset": AB_Dataset,
    "noise": NoiseDataset,
    "NoiseDataset": NoiseDataset,
    "ab+noise": AB_N_Fusion_Dataset,
    "AB_N_Fusion_Dataset": AB_N_Fusion_Dataset,
}
