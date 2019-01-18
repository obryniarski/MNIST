import matplotlib
matplotlib.use('Agg')
print('\n\n\nRunning\n\n\n')


from model import CNN
from preprocessing import get_data
import matplotlib.pyplot as plt
import matplotlib
from sklearn.model_selection import train_test_split
from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import LearningRateScheduler

X, y = get_data(amount=42000)
im_shape = X[0].shape
train_X, val_X, train_y, val_y = train_test_split(X, y, test_size = 0.1)
print(im_shape)
iterate = False
augment = True

if augment:
    datagen = ImageDataGenerator(
            rotation_range=10,
            zoom_range=0.1,
            width_shift_range=0.1,
            height_shift_range=0.1)

annealer = LearningRateScheduler(lambda x: 1e-3 * 0.95 ** x)



if not iterate:

    def train_CNN(data_portion=X.shape[0], epochs=20, ensemble=False):
        model = CNN(im_shape)


        # epochs = 50
        batch = 100
        if augment:
            history = model.fit_generator(datagen.flow(train_X, train_y),
                    epochs=epochs, steps_per_epoch = data_portion // 32,
                    verbose=2, validation_data=[val_X, val_y], callbacks=[annealer])
        else:
            history = model.fit(train_X[:data_portion], train_y[:data_portion], batch, epochs,
                    verbose=2, validation_split=0.3, callbacks=[annealer])
        # print(model.summary())

        if not ensemble:
            # print('saving')
            # model.save('saved_models/my_model.h5')
            # print('done')

            fig = plt.figure(figsize=(15, 5))
            plt.plot(history.history['val_acc'])
            plt.xlabel('epoch')
            plt.ylabel('val_accuracy')
            plt.xlim(0, epochs)
            plt.grid()
            plt.ylim(.97, 1)
            plt.title('Max val_acc: ' + str(max(history.history['val_acc'])))
            fig.savefig('plots/testing.png')


        return model




else:
    iterator = 2
    models = list(CNN(im_shape, iterator))
    styles=[':','-.','--','-',':','-.','--','-',':','-.','--','-']
    fig = plt.figure(figsize=(15, 5))
    names = ['pooling', 'learnable pooling']
    assert len(names) >= iterator, 'Need more names'
    assert len(names) <= iterator, 'Need fewer names'
    assert iterator <= len(styles), 'Need more line styles'

    data_portion = 1500
    epochs = 30
    batch = 100
    history_list = [0] * iterator

    for i in range(iterator):
        print('Training Model : {}'.format(i))
        model = models[i]

        if augment:
            history_list[i] = model.fit_generator(datagen.flow(train_X, train_y),
                    epochs=epochs, steps_per_epoch = data_portion // 32,
                    verbose=0, validation_data=[val_X, val_y], callbacks=[annealer])
            print('Experiment: {0}, Training Accuracy={1:.5f}, Validation Accuracy={2:.5f}'.format(names[i],
                                    max(history_list[i].history['acc']), max(history_list[i].history['val_acc'])))
            print(model.summary())
        else:
            history_list[i] = model.fit(train_X[:data_portion], train_y[:data_portion], batch, epochs,
                verbose=0, validation_split=0.3, callbacks=[annealer])
            print('Experiment: {0}, Training Accuracy={1:.5f}, Validation Accuracy={2:.5f}'.format(names[i],
                                    max(history_list[i].history['acc']), max(history_list[i].history['val_acc'])))


        plt.plot(history_list[i].history['val_acc'], linestyle=styles[i])

    plt.xlabel('epoch')
    plt.ylabel('val_accuracy')
    plt.xlim(0, epochs)
    plt.grid()
    plt.ylim(.97, 1)
    for i in range(iterator):
        names[i] = '{}: {}'.format(names[i], str(max(history_list[i].history['val_acc']))[:6])
    plt.legend(names, loc='upper left')

    fig.savefig('plots/experiment.png')



train_CNN(10000, 15)



def ensemble(num, epochs, data_portion):
    models = [0] * num
    history_list = [0] * num

    for i in range(num):
        models[i] = CNN(im_shape)
        history_list[i] = models[i].fit_generator(datagen.flow(train_X, train_y),
                epochs=epochs, steps_per_epoch = data_portion // 32,
                verbose=2, validation_data=[val_X, val_y], callbacks=[annealer])
        print('CNN: {0}, Training Accuracy={1:.5f}, Validation Accuracy={2:.5f}'.format(i + 1,
                max(history_list[i].history['acc']),
                max(history_list[i].history['val_acc'])))


    return models
