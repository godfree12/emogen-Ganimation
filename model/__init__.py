from .base_model import BaseModel
from .ganimation import GANimationModel
from .stargan import StarGANModel



def create_model(opt):
    if opt.model == "ganimation":
        instance = GANimationModel()
    else:
        instance = BaseModel()
    instance.initialize(opt)
    instance.setup()
    return instance

