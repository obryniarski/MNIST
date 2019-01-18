
from keras.models import Sequential
from keras.layers import Conv2D, Dense, Dropout, BatchNormalization
from keras.layers import MaxPooling2D, Activation, Flatten
import keras

def CNN(input_shape, iteration=None):

    if len(input_shape) < 3:
        input_shape += (1,)
    if not iteration:
        model = Sequential()

        # model.add(Conv2D(16, kernel_size=(5, 5), use_bias=False, padding='same',
        #         input_shape=input_shape))
        # model.add(BatchNormalization())
        # model.add(Activation('relu'))
        # model.add(MaxPooling2D(pool_size=(2, 2)))
        #
        # model.add(Conv2D(32, kernel_size=(5, 5), use_bias=False, padding='same'))
        # model.add(BatchNormalization())
        # model.add(Activation('relu'))
        # model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Conv2D(32, kernel_size = 5, activation='relu', input_shape=input_shape))
        model.add(BatchNormalization())
        model.add(MaxPooling2D(2))
        model.add(BatchNormalization())
        model.add(Flatten())
        model.add(Dense(100, activation='relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.4))
        model.add(Dense(10, activation='softmax'))
        model.compile(loss='categorical_crossentropy',
                optimizer='adam', metrics=['accuracy'])
        return model

        model.add(Conv2D(32, kernel_size = 5, activation='relu', input_shape=input_shape))
        model.add(BatchNormalization())
        model.add(Conv2D(32, kernel_size=5, activation='relu', strides=2, padding='same'))
        model.add(BatchNormalization())

        model.add(Conv2D(64, kernel_size=5, activation='relu'))
        model.add(BatchNormalization())
        model.add(Conv2D(64, kernel_size=5, activation='relu', strides=2, padding='same'))
        model.add(BatchNormalization())
        # model.add(MaxPooling2D(2))
        # model.add(BatchNormalization())

        model.add(Conv2D(96, kernel_size=5, padding='same',activation='relu'))
        model.add(BatchNormalization())


        model.add(Flatten())

        # model.add(Dense(64, activation='relu'))
        # model.add(BatchNormalization())
        # model.add(Dense(44, activation='relu'))
        # model.add(BatchNormalization())
        # model.add(Dropout(0.3))

        model.add(Dropout(0.4))
        model.add(Dense(10, activation='softmax'))

        model.compile(loss='categorical_crossentropy',
                optimizer='adam', metrics=['accuracy'])

        return model
    else:
        models = []
        values = []

        for i in range(iteration):
            model = Sequential()

            model.add(Conv2D(32, kernel_size=5, activation='relu', input_shape=input_shape))
            model.add(BatchNormalization())
            if i == 0:
                model.add(MaxPooling2D(2))
            else:
                model.add(Conv2D(32, kernel_size=5, strides=2, activation='relu', padding='same'))
            model.add(BatchNormalization())

            model.add(Conv2D(64, kernel_size=5, activation='relu'))
            model.add(BatchNormalization())
            if i == 0:
                model.add(MaxPooling2D(2))
            else:
                model.add(Conv2D(64, kernel_size=5, strides=2, activation='relu', padding='same'))
            model.add(BatchNormalization())


            model.add(Conv2D(96, kernel_size=5, padding='same',activation='relu'))
            model.add(BatchNormalization())

            # model.add(MaxPooling2D(pool_size=2)
            # model.add(BatchNormalization())

            model.add(Flatten())

            model.add(Dropout(0.4))

            # model.add(Dense((2 ** (i + 4)) // 2))
            # model.add(Dropout(0.1))

            model.add(Dense(10, activation='softmax'))

            model.compile(loss='categorical_crossentropy',
                    optimizer=keras.optimizers.Adam(lr=0.01), metrics=['accuracy'])

            models.append(model)

        return models
