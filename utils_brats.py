import tensorflow as tf

START_AT = 22
NUM_SLICES = 96
IMG_SIZE = 128
CLASSES = {
    0: 'NOT TUMOR',
    1: 'NECROTIC/CORE',
    2: 'EDEMA',
    3: 'ENHANCING'  # original 4 -> converted into 3 later
}


def preprocess(flair, ce, seg=None, for_pred=False):

    flair = tf.image.resize(flair, [IMG_SIZE, IMG_SIZE])
    ce = tf.image.resize(ce, [IMG_SIZE, IMG_SIZE])
    # stack and slice each channel
    x = tf.stack((
        flair[..., START_AT: START_AT+NUM_SLICES],
        ce[..., START_AT: START_AT+NUM_SLICES]
    ), axis=-1)
    # for y:
    if not for_pred:
        # put channels first
        seg = tf.transpose(seg, [2, 0, 1])
        # slice channels
        seg = seg[START_AT: START_AT+NUM_SLICES]
        # convert into uint8
        seg = tf.cast(seg, tf.uint8)
        # in the dataset, 4 is misplaced with 3,
        #  so we changing 4 to 3.
        seg = tf.where(seg == len(CLASSES),
                       tf.constant(len(CLASSES)-1, seg.dtype),
                       seg)
        # one hot encoding
        y = tf.one_hot(seg, len(CLASSES))
        # resize
        y = tf.image.resize(y, (IMG_SIZE, IMG_SIZE))
        y = tf.transpose(y, [1, 2, 0, 3])
        # flatten
        return tf.reshape(x, [-1]), tf.reshape(y, [-1])
    return x


def parse_records(tfdataset):
    features = {
        "x": tf.io.FixedLenFeature(IMG_SIZE*IMG_SIZE*NUM_SLICES*2,
                                   tf.float32),
        "y": tf.io.FixedLenFeature(IMG_SIZE*IMG_SIZE*NUM_SLICES*len(CLASSES),
                                   tf.float32)
    }
    parsed_example = tf.io.parse_example(tfdataset, features)
    x = tf.reshape(parsed_example['x'],
                   [-1, IMG_SIZE, IMG_SIZE, NUM_SLICES, 2])
    y = tf.reshape(
        parsed_example['y'], [-1, IMG_SIZE, IMG_SIZE, NUM_SLICES, len(CLASSES)])
    x, y = tf.transpose(x, [0, 3, 1, 2, 4]), tf.transpose(y, [0, 3, 1, 2, 4])
    return x, y


def dice_coef(y_true, y_pred, smooth=1.0):
    class_num = len(CLASSES)
    for i in range(class_num):
        y_true_f = tf.reshape(y_true[..., i], [-1])
        y_pred_f = tf.reshape(y_pred[..., i], [-1])
        intersection = tf.reduce_sum(y_true_f * y_pred_f)
        loss = ((2.0 * intersection + smooth) / (tf.reduce_sum(y_true_f) +
                                                 tf.reduce_sum(y_pred_f) + smooth))
        if i == 0:
            total_loss = loss
        else:
            total_loss = total_loss + loss
    total_loss = total_loss / class_num
    return total_loss


def dice_coef_necrotic(y_true, y_pred, epsilon=1e-6):
    intersection = tf.reduce_sum(tf.abs(y_true[..., 1] * y_pred[..., 1]))
    return (2.0 * intersection) / (tf.reduce_sum(tf.square(y_true[..., 1])) +
                                   tf.reduce_sum(tf.square(y_pred[..., 1])) + epsilon)


def dice_coef_edema(y_true, y_pred, epsilon=1e-6):
    intersection = tf.reduce_sum(tf.abs(y_true[..., 2] * y_pred[..., 2]))
    return (2.0 * intersection) / (tf.reduce_sum(tf.square(y_true[..., 2])) +
                                   tf.reduce_sum(tf.square(y_pred[..., 2])) + epsilon)


def dice_coef_enhancing(y_true, y_pred, epsilon=1e-6):
    intersection = tf.reduce_sum(tf.abs(y_true[..., 3] * y_pred[..., 3]))
    return (2.0 * intersection) / (tf.reduce_sum(tf.square(y_true[..., 3])) +
                                   tf.reduce_sum(tf.square(y_pred[..., 3])) + epsilon)


def precision(y_true, y_pred, epsilon=1e-6):
    true_positives = tf.reduce_sum(
        tf.round(tf.clip_by_value(y_true * y_pred, 0, 1)))
    predicted_positives = tf.reduce_sum(
        tf.round(tf.clip_by_value(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + epsilon)
    return precision


def sensitivity(y_true, y_pred, epsilon=1e-6):
    true_positives = tf.reduce_sum(
        tf.round(tf.clip_by_value(y_true * y_pred, 0, 1)))
    possible_positives = tf.reduce_sum(
        tf.round(tf.clip_by_value(y_true, 0, 1)))
    return true_positives / (possible_positives + epsilon)


def specificity(y_true, y_pred, epsilon=1e-6):
    true_negatives = tf.reduce_sum(
        tf.round(tf.clip_by_value((1 - y_true) * (1 - y_pred), 0, 1)))
    possible_negatives = tf.reduce_sum(
        tf.round(tf.clip_by_value(1 - y_true, 0, 1)))
    return true_negatives / (possible_negatives + epsilon)
