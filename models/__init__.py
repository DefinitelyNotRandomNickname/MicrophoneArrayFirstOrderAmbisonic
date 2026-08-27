from models.conformer import Conformer
from models.tfgridnet import TFGridNet
from models.unet import UNet

MODELS = {
    "Conformer": Conformer,
    "TFGridNet": TFGridNet,
    "UNet": UNet,
}
