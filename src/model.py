# Import necessary libraries
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, BatchNormalization, Activation, GlobalAveragePooling2D
from keras.optimizers import Adam


class CNNModel:
    def __init__(self, input_shape, num_classes):
        self.model = CNNModel.build_model(input_shape, num_classes)

    @staticmethod
    def build_model(input_shape, num_classes):
        model = Sequential()

        model.add(Conv2D(32, (3, 3), padding='same', strides=(1, 1), input_shape=input_shape, use_bias=False))
        model.add(BatchNormalization(scale=False))
        model.add(Activation('relu'))

        model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))

        model.add(Conv2D(64, (3, 3), padding='same', strides=(1, 1), input_shape=(55, 55, 32), use_bias=False))
        # model.add(Conv2D(64, (3, 3), padding='same', strides=(1, 1), use_bias=False))
        model.add(BatchNormalization(scale=False))
        model.add(Activation('relu'))

        model.add(Conv2D(96, (3, 3), padding='same', strides=(1, 1), input_shape=(55, 55, 64), use_bias=False))
        model.add(BatchNormalization(scale=False))
        model.add(Activation('relu'))

        model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))

        model.add(Conv2D(128, (3, 3), padding='same', strides=(1, 1), input_shape=(27, 27, 96), use_bias=False))
        model.add(BatchNormalization(scale=False))
        model.add(Activation('relu'))

        model.add(Conv2D(256, (3, 3), padding='same', strides=(1, 1), input_shape=(27, 27, 128), use_bias=False))
        model.add(BatchNormalization(scale=False))
        model.add(Activation('relu'))

        model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))

        model.add(Conv2D(384, (3, 3), padding='same', strides=(1, 1), input_shape=(13, 13, 256), use_bias=False))
        model.add(BatchNormalization(scale=False))
        model.add(Activation('relu'))

        model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))

        model.add(GlobalAveragePooling2D())

        model.add(Dense(64, activation='softmax'))
        model.add(Dense(num_classes, activation='softmax'))

        model.summary()
        # keras.utils.plot_model(model, show_shapes=True)

        # Compile the model
        model.compile(optimizer='adam',
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])

        return model

    def summary(self):
        self.model.summary()

    def train(self, train_dataset, validation_dataset, epochs, batch_size, cp_callback):
        history = self.model.fit(train_dataset,
                                 epochs=epochs,
                                 batch_size=batch_size,
                                 validation_data=validation_dataset,
                                 callbacks=[cp_callback])
        return history

    def evaluate(self, test_dataset):
        return self.model.evaluate(test_dataset)

    def predict(self, test_dataset):
      return self.model.predict(test_dataset)

    def save(self, path):
      return self.model.save(path)
