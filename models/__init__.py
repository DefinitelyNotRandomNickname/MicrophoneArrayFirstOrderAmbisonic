from models.conformer import Conformer
from models.mamba import Mamba
from models.spatialnet import SpatialNet
from models.tfgridnet import TFGridNet
from models.unet import UNet

MODELS = {
    "Conformer": Conformer,
    "Mamba": Mamba,
    "SpatialNet": SpatialNet,
    "TFGridNet": TFGridNet,
    "UNet": UNet,
}
