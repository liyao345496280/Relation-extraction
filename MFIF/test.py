from train import Framework
from config import Config
import torch
import numpy as np

seed = 1234
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
np.random.seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

config = Config()


fw = Framework(config)

model_name="globalpointerlinker_adamw_nyt.pt"

fw.testall(model_name)

