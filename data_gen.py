import csv
import numpy as np
import cv2
import matplotlib.image as mpimg
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

class DataGen:
    def __init__(self, log_file="./data/driving_log.csv", batch_size=128):
        self._train = []
        with open(log_file) as csvf:
            reader = csv.reader(csvf)
            next(reader)
            for line in reader:
                self._train.append(line)
        self._train = np.array(self._train)
        self._train, self._validation = train_test_split(self._train, test_size=.2)
        self._batch_size = batch_size
        self._views = ['center', 'left', 'right']

    def samples_per_epoch(self):
        return self._train.shape[0] * 4

    def next_train(self):
            num_samples = self._train.shape[0]
            while True:
                shuffle(self._train)
                for start in range(0, num_samples, self._batch_size):
                    end = start + self._batch_size
                    angles = [ float(x) for x in self._train[start:end, 3] ]
                    correction = .20
                    for view in self._views:
                        if view == 'center':
                            images = [ self._rand_brightness(x) for x in self._train[start:end, 0] ]
                            truth = np.array(angles)
                            train_out = shuffle(np.array(images), truth)
                            yield train_out
                            images = [np.fliplr(img) for img in images]
                            truth = -truth
                            train_out = shuffle(np.array(images), truth)
                            yield train_out
                        elif view == 'left':
                            images = [ self._rand_brightness(x) for x in self._train[start:end, 1] ]
                            truth = np.array(angles) + correction
                            train_out = shuffle(np.array(images), truth)
                            yield train_out
                        else: #view == 'right'
                            images = [ self._rand_brightness(x) for x in self._train[start:end, 2] ]
                            truth = np.array(angles) - correction
                            train_out = shuffle(np.array(images), truth)
                            yield train_out


    def next_valid(self):
            num_samples = self._validation.shape[0]
            while True:
                shuffle(self._validation)
                for start in range(0, num_samples, self._batch_size):
                    end = start + self._batch_size
                    images = [ mpimg.imread("./data/" + x) for x in self._validation[start:end, 0] ]
                    angles = [ float(x) for x in self._validation[start:end, 3] ]
                    valid_out = shuffle(np.array(images), np.array(angles))
                    yield valid_out

    def _rand_brightness(self, fname):
        img = cv2.imread("./data/" + fname.strip())
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        mask = np.zeros_like(hsv[:, :, 2]) + 255
        hsv[:,:,2] = np.minimum((np.random.rand() -.5) * hsv[:, :, 2], mask)
        return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
