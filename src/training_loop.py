from src.model import CNNModel
import keras.utils
import matplotlib.pyplot as plt
import datetime
import os

# Parameters
k_folds=5
batch_size=32
image_size= (112,112)
image_shape=(112,112,3)
num_classes=5
epochs=40

for fold in range(k_folds):
    print(f'Fold {fold + 1} begins...')
    model = CNNModel(input_shape=image_shape, num_classes=num_classes)

    # Getting paths for each fold
    train_dir = f'./data/kfold_dataset/fold_{fold+1}/train'
    val_dir = f'./data/kfold_dataset/fold_{fold+1}/validation'
    test_dir = f'./data/kfold_dataset/fold_{fold+1}/test'

    # Loading train, validation and test datasets for each fold
    train_dataset = keras.utils.image_dataset_from_directory(
        directory=train_dir,
        labels='inferred',
        label_mode='categorical',
        batch_size=batch_size,
        image_size=image_size
    )

    val_dataset = keras.utils.image_dataset_from_directory(
        directory=val_dir,
        labels='inferred',
        label_mode='categorical',
        batch_size=batch_size,
        image_size=image_size
    )

    test_dataset = keras.utils.image_dataset_from_directory(
        directory=test_dir,
        labels='inferred',
        label_mode='categorical',
        batch_size=batch_size,
        image_size=image_size
    )
    #----------------------------------------------------------

    unique = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
    # Saving checkpoints
    checkpoint_path=f'./models/saved_weights/cnn_model_epoch_{epochs}_fold_{fold+1}_{unique}.ckpt'
    checkpoint_dir =os.path.dirname(checkpoint_path)
    cp_callback = keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                 save_weights_only=True,
                                                 verbose=1)

    # Training
    history = model.train(train_dataset, val_dataset, epochs, batch_size, cp_callback)

    # Plotting accuracy
    plt.plot(history.history['accuracy'], label='accuracy')
    plt.plot(history.history['val_accuracy'], label='validation accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend(loc='lower right')
    plt.title(f'Accuracy graph for fold {fold+1}')
    plt.plot()
    plt.savefig(f'./results/accuracy_epoch_{epochs}_fold_{fold+1}_{unique}.png')
    plt.show()

    # Plotting loss
    plt.plot(history.history['loss'], label='loss')
    plt.plot(history.history['val_loss'], label='validation loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(loc='lower right')
    plt.title(f'Loss graph for fold {fold+1}')
    plt.plot()
    plt.savefig(f'./results/loss_epoch_{epochs}_fold_{fold+1}_{unique}.png')
    plt.show()

    # Prediction
    loss, accuracy = model.evaluate(test_dataset)
    print(f'Test Loss: {loss}')
    print(f'Test Accuracy: {accuracy}')

    # Saving entire model
    model.save(f'./models/saved_models/cnn_model_epoch_{epochs}_fold_{fold+1}_{unique}.keras')

    print(f'Fold {fold+1} ends...')