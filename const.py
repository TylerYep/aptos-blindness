RUN_ON_GPU = False
RUN_ID = 'baseline'
# CONTINUE_FROM = f'save/{RUN_ID}/weights_1.pth'
CONTINUE_FROM = None
LAST_SAVE = 0
MODELS = ('xception',)
CURR_MODEL = MODELS[0]

DATA_PATH = 'data/'
if RUN_ON_GPU:
    BATCH_SIZE = 32
    TRAIN_CSV = DATA_PATH + 'train.csv'
else:
    BATCH_SIZE = 4
    TRAIN_CSV = DATA_PATH + 'train-small.csv'

LEARNING_RATE = 3e-4
MAX_STEPS_PER_EPOCH = 15000
NUM_EPOCHS = 50

INPUT_SHAPE = (229, 229)
DEV_CSV = DATA_PATH + 'dev.csv'
TEST_CSV = DATA_PATH + 'test.csv'
