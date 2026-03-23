from .network.ac_only import OnlyAC
from .network.bc_only import OnlyBC
from .network.acbc import ACBC
from .generator.generator import GeneratorDBMIF
from .generator.tri_generator import Generator as TriGenerator
from .generator.mono_generator import MonoGenerator
from .generator.dual_generator import DualGenerator
from .discriminator.discriminator import (
    DiscriminatorDBMIFMultiScales as Discriminator,
)
from .contrast.FCN.FCN import FCN
from .contrast.DCCRN.DCCRN import DCCRN as DenseConnetCRN
from .contrast.DenGCAN.DenGCAN import DenGCAN
from .contrast.MMINet.MMINet import MMINet
from .contrast.Conformer.Comformer import ConformerBasedEnhancer as Conformer
from .contrast.GaGNet.GaGNet import GaGNetModule as GaGNet

MODEL_MAP = {
    "OnlyAC": OnlyAC,
    "OnlyBC": OnlyBC,
    "ACBC": ACBC,
    "GeneratorDBMIF": GeneratorDBMIF,
    "TriGenerator": TriGenerator,
    "MonoGenerator": MonoGenerator,
    "DualGenerator": DualGenerator,
    "Discriminator": Discriminator,
    "FCN": FCN,
    "DenseConnetCRN": DenseConnetCRN,
    "DenGCAN": DenGCAN,
    "MMINet": MMINet,
    "Conformer": Conformer,
    "GaGNet": GaGNet,
}
