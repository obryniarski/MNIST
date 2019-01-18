from keras.models import load_model
from preprocessing import get_data
from train import ensemble
import keras
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# model = load_model('saved_models/my_model.h5')

def submit(model, do_ensemble=False, num_ensembles=3):
    test_X = get_data(training=False)
    print(test_X.shape)
    # plt.imshow(test_X[32].reshape((28, 28)))
    # plt.show()

    if not do_ensemble:

        results = list(map(lambda pred: np.argmax(pred), model.predict(test_X)))
        results = np.array(results)
        print(results.shape)
        print(results[:10])
        submission = pd.DataFrame({'Label': results}, list(range(1, 28001)))
        print(submission.head())

        submission.to_csv('submissions/submission.csv', index_label='ImageId')

    if do_ensemble:
        models = ensemble(num_ensembles, 30, 2000)

        results = np.zeros( (test_X.shape[0], 10))
        for i in range(len(models)):
            results = results + models[i].predict(test_X)

        results = np.argmax(results, axis=1)
        print(results.shape)
        print(results[:10])
        submission = pd.DataFrame({'Label': results}, list(range(1, 28001)))
        # submission.to_csv('submissions/ensemble_prediction.csv', index_label='ImageID')
        submission.to_csv('ensemble_prediction.csv', index_label='ImageID')



submit(None, True, 5)
