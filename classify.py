import cv2
import pickle
from skimage import img_as_ubyte


def crop_symbol(symbol):

    y, x = symbol.shape

    min_y = 0
    max_y = y - 1
    min_x = 0
    max_x = x - 1

    for i in range(y):
        if symbol[i].any():
            min_y = i
            break

    for i in range(y-1, -1, -1):
        if symbol[i].any():
            max_y = i
            break

    for i in range(x):
        if symbol[:, i].any():
            min_x = i
            break

    for i in range(x-1, -1, -1):
        if symbol[:, i].any():
            max_x = i
            break
    cropped_symbol = symbol[min_y:max_y, min_x:max_x]
    return cropped_symbol


target_img_size = (32, 32)


def resize_img(img):
    resized_img = cv2.resize(img, target_img_size)
    return resized_img


def extract_hog_features(img):

    win_size = (32, 32)
    cell_size = (4, 4)
    block_size_in_cells = (2, 2)

    block_size = (block_size_in_cells[1] * cell_size[1],
                  block_size_in_cells[0] * cell_size[0])
    block_stride = (cell_size[1], cell_size[0])
    nbins = 9  # Number of orientation bins
    hog = cv2.HOGDescriptor(win_size, block_size,
                            block_stride, cell_size, nbins)
    h = hog.compute(img)
    h = h.flatten().flatten()

    return h


def extract_features(img, resize=True):

    if resize:
        img = resize_img(img)

    features = extract_hog_features(img)
    return features


def classify(symbols):
    model = None
    with open('model_pickle1', 'rb') as f:
        model = pickle.load(f)
    note_predictions = []
    for row in symbols:
        row_predictions = []
        for symbol in row:
            symbol = img_as_ubyte(crop_symbol(symbol))
            features = extract_features(symbol)
            svm_prediction = model.predict([features])
            row_predictions.append(svm_prediction[0])
        note_predictions.append(row_predictions)

    return note_predictions
