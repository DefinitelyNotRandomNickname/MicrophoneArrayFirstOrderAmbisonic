from models.conformer import Conformer
from models.mamba import Mamba
from models.tfgridnet import TFGridNet
from models.unet import UNet

MODELS = {
    "Conformer": Conformer,
    "Mamba": Mamba,
    "TFGridNet": TFGridNet,
    "UNet": UNet,
}
