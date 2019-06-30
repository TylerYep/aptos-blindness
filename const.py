RUN_ON_GPU = True
RUN_ID = 'selfattn'
# CONTINUE_FROM = f'save/{RUN_ID}/weights_395000.0.pth'
CONTINUE_FROM = None
MODELS = ('resnet50')
CURR_MODEL = MODELS[0]

NUM_CLASSES = 4
DATA_PATH = 'data/' #'../input/aptos2019-blindness-detection/'
if RUN_ON_GPU:
    BATCH_SIZE = 16
    TRAIN_CSV = DATA_PATH + 'train.csv'
else:
    BATCH_SIZE = 4
    TRAIN_CSV = DATA_PATH + 'train-subset.csv'

LEARNING_RATE = 3e-4
MAX_STEPS_PER_EPOCH = 15000
NUM_EPOCHS = 500
LOG_FREQ = 5000
PLT_FREQ = 100
NUM_TOP_PREDICTS = 20

INPUT_SHAPE = (299, 299)
DEV_CSV = DATA_PATH + 'dev.csv'
TEST_CSV = DATA_PATH + 'test.csv'
