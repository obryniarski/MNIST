import pandas as pd
import numpy as np
import cv2
import keras
from keras.preprocessing.image import ImageDataGenerator
from imgaug import augmenters as iaa

def get_data(augment_data=False, amount=None, training=True):

    # read data
    if training:
        try:
            train_data = pd.read_csv('/floyd/input/data/train.csv')
        except:
            train_data = pd.read_csv('/Users/OliverBryniarski 1/Desktop/datasets/mnist_data/train.csv')
        if not amount:
            amount = train_data.shape[0]
        X = np.empty((amount, 28, 28))
        y = np.empty(amount)
        for img in range(amount):
            X[img] = np.array(train_data.loc[img][1:]).reshape(28,28) / 255
            y[img] = train_data.loc[img][0]


        y = keras.utils.to_categorical(y, 10)
        X = X.reshape(X.shape + (1,))

        #preprocess
        if augment_data:
            X, y = augment((X, y))

        return X, y

    else:
        try:
            train_data = pd.read_csv('/floyd/input/data/test.csv')
        except:
            train_data = pd.read_csv('/Users/OliverBryniarski 1/Desktop/datasets/mnist_data/test.csv')
        test_X = np.empty((28000, 28, 28))
        for img in range(28000):
            test_X[img] = np.array(train_data.loc[img]).reshape(28,28) / 255
        test_X = test_X.reshape(test_X.shape + (1,))

        return test_X


def augment(data, multiples=1):
    orig_X, orig_y = data
    # aug = iaa.SomeOf(2, [
    #                      iaa.Affine(rotate=(-30, 30)),
    #                      # iaa.Affine(translate_px={'x': (-7, 7), 'y': (-7, 7)}),
    #                      iaa.Affine(shear=(-16, 16)),
    #                      iaa.Affine(scale={'x': (0.7, 1), 'y': (0.5, 1)}),
    #                  ], random_order=True)



#                        | add new X to X|             | double up y to match new X |
    # return (np.append(aug.augment_images(X), X, axis=0), np.append(y, y, axis=0))
    # datagen = ImageDataGenerator(
    #         rotation_range=10,
    #         zoom_range=0.1,
    #         width_shift_range=0.1,
    #         height_shift_range=0.1)
    #
    # more_X, more_y = list(datagen.flow(orig_X, orig_y, batch_size=orig_X.shape[0]))[0]
    # if multiples == 1:
    #     X, y = np.append(orig_X, more_X, axis=0), np.append(orig_y, more_y, axis=0)
    #     print('Shape of X: {} \nShape of y: {}'.format(X.shape, y.shape))
    #     return X, y
    #
    # for i in range(multiples - 1):
    #     more_data = list(datagen.flow(orig_X, orig_y, batch_size=orig_X.shape[0]))[0]
    #     more_X, more_y = np.append(more_data[0], more_X, axis=0), np.append(more_data[1], more_y, axis=0)
    #
    # X, y = np.append(orig_X, more_X, axis=0), np.append(orig_y, more_y, axis=0)
    #
    # print('Shape of X: {} \nShape of y: {}'.format(X.shape, y.shape))
    # return X, y
    return None





def display_random(augment=False, sample=None):
    if not sample:
        X, y = get_data(augment)
    else:
        X, y = get_data(augment, sample)

    sample = len(X)
    if augment:
        rand_idx = np.random.randint(0, sample//2)
    else:
        rand_idx = np.random.randint(0, sample)

    print(y.shape)
    print(y[rand_idx])

    cv2.imshow('test', X[rand_idx])
    cv2.waitKey(0)

# display_random(True, 100)
