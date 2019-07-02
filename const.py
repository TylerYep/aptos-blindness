RUN_ON_GPU = False
RUN_ID = 'baseline'
# CONTINUE_FROM = f'save/{RUN_ID}/weights_1.pth'
CONTINUE_FROM = None
LAST_SAVE = 0
MODELS = ('resnet101',)
CURR_MODEL = MODELS[0]

NUM_CLASSES = 5
DATA_PATH = 'data/' #'../input/aptos2019-blindness-detection/'
if RUN_ON_GPU:
    BATCH_SIZE = 16
    TRAIN_CSV = DATA_PATH + 'train.csv'
else:
    BATCH_SIZE = 4
    TRAIN_CSV = DATA_PATH + 'train-subset.csv'

LEARNING_RATE = 3e-4
MAX_STEPS_PER_EPOCH = 15000
NUM_EPOCHS = 50
LOG_FREQ = 5000
PLT_FREQ = 100
NUM_TOP_PREDICTS = 20

INPUT_SHAPE = (224, 224)
DEV_CSV = DATA_PATH + 'dev.csv'
TEST_CSV = DATA_PATH + 'test.csv'
