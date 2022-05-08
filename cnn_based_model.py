# %%
from methods import *
from matplotlib.pyplot import imshow
from sklearn.model_selection import train_test_split

from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
# plt.style.use('dark_background')

# 0 - Tumor
# 1 - Normal


path_tumor = 'brain_tumor_dataset/yes'
path_normal = 'brain_tumor_dataset/no'


def main():

    data_tumor, labels_tumor = read_and_resize_images(path_tumor, 128, 0)
    data_normal, labels_normal = read_and_resize_images(path_normal, 128, 1)

    data = np.array([*data_tumor, *data_normal])
    labels = np.array([*labels_tumor, *labels_normal])

    labels = labels.reshape(labels.shape[0], 2)

    data.shape
    labels.shape

    x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2,
                                                        shuffle=True, random_state=0)

    # %%
    model = Sequential()

    model.add(Conv2D(32, kernel_size=(2, 2),
                     input_shape=(128, 128, 3), padding='Same'))
    model.add(Conv2D(32, kernel_size=(2, 2),
              activation='relu', padding='Same'))

    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(64, kernel_size=(2, 2),
                     activation='relu', padding='Same'))
    model.add(Conv2D(64, kernel_size=(2, 2),
                     activation='relu', padding='Same'))

    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())

    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(2, activation='softmax'))

    model.compile(loss="categorical_crossentropy",
                  optimizer='Adam', metrics=['accuracy'])
    print(model.summary())

    # %%
    y_train.shape

    # %%
    history = model.fit(x_train, y_train, epochs=30,
                        batch_size=40,
                        verbose=1, validation_data=(x_test, y_test))

    # %%
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Test', 'Validation'], loc='upper right')
    plt.show()

    # %%

    def names(number):
        if number == 0:
            return 'Its a Tumor'
        else:
            return 'No, Its not a tumor'

    # %%
    img = Image.open("brain_tumor_dataset/no/N19.JPG")
    x = np.array(img.resize((128, 128)))
    x = x.reshape(1, 128, 128, 3)
    res = model.predict_on_batch(x)
    classification = np.where(res == np.amax(res))[1][0]
    imshow(img)
    print(str(res[0][classification]*100) +
          '% Confidence This Is ' + names(classification))

    # %%
    img = Image.open(
        r"brain_tumor_dataset/yes/Y3.jpg")
    x = np.array(img.resize((128, 128)))
    x = x.reshape(1, 128, 128, 3)
    res = model.predict_on_batch(x)
    classification = np.where(res == np.amax(res))[1][0]
    imshow(img)
    print(str(res[0][classification]*100) +
          '% Confidence This Is A ' + names(classification))


if __name__ == '__main__':
    main()
