import numpy as np
from utils.metrics.confusionmatrix import ConfusionMatrix


class IoU:
    """Computes the intersection over union (IoU) per class and corresponding
    mean (mIoU).

    Intersection over union (IoU) is a common evaluation metric for semantic
    segmentation. The predictions are first accumulated in a confusion matrix
    and the IoU is computed from it as follows:

        IoU = true_positive / (true_positive + false_positive + false_negative).

    Keyword arguments:
    - num_classes (int): number of classes in the classification problem
    - normalized (boolean, optional): Determines whether or not the confusion
    matrix is normalized or not. Default: False.
    - ignore_index (int or iterable, optional): Index of the classes to ignore
    when computing the IoU. Can be an int, or any iterable of ints.
    """

    def __init__(self, num_classes, normalized=False, ignore_index=None):
        super().__init__()
        self.conf_metric = ConfusionMatrix(num_classes, normalized)

        if ignore_index is None:
            self.ignore_index = None
        elif isinstance(ignore_index, int):
            self.ignore_index = (ignore_index,)
        else:
            try:
                self.ignore_index = tuple(ignore_index)
            except TypeError:
                raise ValueError("'ignore_index' must be an int or iterable")

    def reset(self):
        self.conf_metric.reset()

    def add(self, predicted, target):
        """Adds the predicted and target pair to the IoU metric.

        Keyword arguments:
        - predicted (Tensor): Can be a (N, K, H, W) tensor of
        predicted scores obtained from the model for N examples and K classes,
        or (N, H, W) tensor of integer values between 0 and K-1.
        - target (Tensor): Can be a (N, K, H, W) tensor of
        target scores for N examples and K classes, or (N, H, W) tensor of
        integer values between 0 and K-1.

        """
        # Dimensions check
        #rint('predicted', predicted)
        #print('predicted shape', predicted.shape)
        #print('target', target)
        #print('target shape', target.shape)
        assert predicted.size(0) == target.size(0), \
            'number of targets and predicted outputs do not match'
        assert predicted.dim() == 1 or predicted.dim() == 2 or predicted.dim() == 3 or predicted.dim() == 4, \
            "predictions must be of dimension (N) or (N, K) or (N, H, W) or (N, K, H, W)"
        assert target.dim() == 1 or target.dim() == 2 or target.dim() == 3 or target.dim() == 4, \
            "targets must be of dimension (N) or (N, K) or (N, H, W) or (N, K, H, W)"

        # If the tensor is in categorical format convert it to integer format
        if predicted.dim() == 2 or predicted.dim() == 4:
            _, predicted = predicted.max(1)
        if target.dim() == 2 or target.dim() == 4:
            _, target = target.max(1)

        self.conf_metric.add(predicted.view(-1), target.view(-1))

    def value(self, eps = 1e-7):
        """Computes the IoU and mean IoU.

        The mean computation ignores NaN elements of the IoU array.

        Returns:
            Tuple: (IoU, mIoU). The first output is the per class IoU,
            for K classes it's numpy.ndarray with K elements. The second output,
            is the mean IoU.
        """
        conf_matrix = self.conf_metric.value()
        if self.ignore_index is not None:
            conf_matrix[:, self.ignore_index] = 0
            conf_matrix[self.ignore_index, :] = 0
        true_positive = np.diag(conf_matrix)
        A = np.sum(conf_matrix, 1)
        B = np.sum(conf_matrix, 0)
        false_positive = B - true_positive
        false_negative = A - true_positive

        # Just in case we get a division by 0, ignore/hide the error
        with np.errstate(divide='ignore', invalid='ignore'):
            iou = true_positive / (true_positive + false_positive + false_negative)
            f1 = true_positive / (true_positive + 0.5 * (false_positive + false_negative))
            dice = (2 * true_positive) / (A + B + eps)

        pixel_accuracy = np.sum(true_positive) / np.sum(conf_matrix)

        return iou, np.nanmean(iou), f1, np.nanmean(f1), np.nanmean(dice), pixel_accuracy

