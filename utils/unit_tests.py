import numpy as np
from utils import model_utils


def check_nan_in_model(model, logger):
    num_nan = model_utils.count_nan_parameters(model).item()
    if num_nan > 0:
        logger.error('Num NaN model params:', num_nan, '/', model_utils.count_parameters(model),
                     'Try turning on torch.autograd.set_detect_anomaly(True).')


def compare_train_val(train_dataset, val_dataset, train_cropped=False):
    for item_train in train_dataset:
        item_train_formatted = item_train.rsplit('_', 1)
        if train_cropped:
            assert len(item_train_formatted) == 2,\
                'ERROR: Requested unit test using train_cropped=True on noncropped name'
            item_train_name = item_train_formatted[0]
        else:
            item_train_name = item_train
        for item_val in val_dataset:
            assert not np.array_equal(item_train_name, item_val), \
                'ERROR: Validation dataset contains the same data as train dataset for: {}'.format(item_train)