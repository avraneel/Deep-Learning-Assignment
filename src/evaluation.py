import keras
from sklearn.metrics import confusion_matrix, classification_report
import os
import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score


def get_label(image, label):
    return label


k_folds = 5
p1 = '../models/saved_models/'
uid = ['20231126163451', '20231126164022', '20231126164727', '20231126165412', '20231126165955']

batch_size = 32
image_size = (112, 112)

for fold in range(k_folds):

    test_dir = f'../data/kfold_dataset/fold_{fold + 1}/test'

    test_dataset = keras.utils.image_dataset_from_directory(
        directory=test_dir,
        labels='inferred',
        label_mode='categorical',
        batch_size=batch_size,
        image_size=image_size,

    )
    print(f'RESULTS:')
    print(f'---------------------------------')
    print(f'Model name - cnn_model_epoch_40_fold_{fold + 1}_{uid[fold]}.keras')
    print('----------------------------------')
    model = keras.models.load_model(os.path.join(p1, f'cnn_model_epoch_40_fold_{fold + 1}_{uid[fold]}.keras'))

    predictions = np.array([])
    labels = np.array([])
    for x, y in test_dataset:
        predictions = np.concatenate([predictions, np.argmax(model.predict(x, verbose=0), axis=-1)])
        labels = np.concatenate([labels, np.argmax(y.numpy(), axis=-1)])

    cmsk = confusion_matrix(labels, predictions)

    print('Confusion matrix: ')
    print(cmsk)

    accuracy_sk = accuracy_score(labels, predictions)
    _, accuracy = model.evaluate(test_dataset, verbose=0)

    print(f'Accuracy from sklearn accuracy score: {accuracy_sk}')
    print(f'Accuracy from model.evaluate(): {accuracy}')

    correct_labels = np.where(predictions == labels)[0]
    wrong_labels = np.where(predictions != labels)[0]

    print(f'Correct Labels: {len(correct_labels)}')
    print(f'Wrong Labels: {len(wrong_labels)}')

    class_report = classification_report(labels, predictions)
    print('Classification Report:\n', class_report)

    print(f'Fold {fold + 1} evaluation ends.')
    print('------------------------------------')
    # break