import numpy as np
import torch
torch.manual_seed(0)
np.random.seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

RUN_ON_GPU = True
RUN_ON_COLAB = True
GIT_PATH = '/content' if RUN_ON_COLAB else '' # TODO

USE_TENSORBOARD = False
RUN_ID = ''
CONTINUE_FROM = None # f'save/{RUN_ID}/weights_1.pth'
LAST_SAVE = 0
# MODELS = ('xception',)
# CURR_MODEL = MODELS[0]

# DATA_PATH = '../gdrive/My Drive/Colab Notebooks/' if RUN_ON_COLAB else 'data/'
DATA_PATH = '/content/' if RUN_ON_COLAB else 'data/'

if RUN_ON_GPU:
    BATCH_SIZE = 32
    TRAIN_CSV = 'data/train.csv'
else:
    BATCH_SIZE = 6
    TRAIN_CSV = 'data/train-small.csv'

LEARNING_RATE = 1e-3
MAX_STEPS_PER_EPOCH = 150000
NUM_EPOCHS = 1000

INPUT_SHAPE = (256, 256)
DEV_CSV = 'data/dev.csv'
TEST_CSV = 'data/test.csv'
