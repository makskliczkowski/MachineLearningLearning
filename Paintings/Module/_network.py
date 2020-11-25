from keras import backend as K
from tensorflow.python.client import device_lib
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
import pandas as pd
from keras.layers import Activation, Dropout, Flatten, Dense, GlobalAveragePooling2D
from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from keras.models import load_model
import keras
import matplotlib.pyplot as plt
import matplotlib
from keras.applications import VGG19
import numpy as np


def printHistory(history):
    # val_acc = history.history['val_acc']
    plt.plot(history.history['val_accuracy'])
    plt.plot(history.history['loss'])
    # plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Loss and accuracy')
    plt.legend()
    plt.show()


def createModel(df, dir, savename,columns,types_num, epo=10, batch=32):
    DATASET_LOCATION = dir
    BATCH_SIZE = batch
    IMAGE_SIZE = (128, 128)
    INPUT_SHAPE = (128, 128, 3)
    EPOCHS = epo
    conv_base = VGG19(weights='imagenet', include_top=False, input_shape=(128, 128, 3))
    conv_base.trainable = False
    # Tworzymy bazę na podstawie conv modelu bez górnego klasyfikatora

    # Instantiating a Convolutional Neural Network (CNN) Classifier
    model = Sequential()
    # biggest -----------
    # --------vectoor
    for layer in conv_base.layers:
        layer.trainable = False
    # ------frozen base-------------
    model.add(conv_base)

    # model.add(Conv2D(filters=32,kernel_size=3,padding='same',activation='relu', input_shape=INPUT_SHAPE))
    # model.add(Conv2D(filters=32,kernel_size=3,padding='same',activation='relu'))
    # model.add(MaxPooling2D(2, 2))
    # model.add(Conv2D(64, (3, 3), activation= 'relu',padding='same'))
    # model.add(Conv2D(64, (3, 3), activation= 'relu',padding='same'))
    # model.add(MaxPooling2D(2, 2))

    # model.add(Conv2D(128, (3, 3), activation='relu'))
    # model.add(MaxPooling2D(2, 2))
    # conv_base.summary()

    model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors
    # model.add(GlobalAveragePooling2D())
    model.add(Dense(512, activation='relu'))
    model.add(Dense(256, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(types_num, activation='softmax'))

    model.compile(
        loss=keras.losses.categorical_crossentropy,
        optimizer=keras.optimizers.adam(),
        metrics=["accuracy"],
    )
    print('Initialized model\n')
    # separate in training and testing
    train_df, test_df = train_test_split(df, test_size=0.35, random_state=40)
    # data augmentation - to provide more samples
    train_datagen = ImageDataGenerator(
        rescale=1. / 255,
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest',
    )
    # cannot change validation data!
    test_datagen = ImageDataGenerator(rescale=1. / 255)
    print('Created data augmentation method, now we have more data! Cool huh?\n')
    # read files of a difrectory using flow from dataframe
    # FIRST FOR TRAIN SECOND FOR TEST
    try:
        train_generator = train_datagen.flow_from_dataframe(
            train_df,
            DATASET_LOCATION,
            x_col=columns[0],
            y_col=columns[1],
            target_size=IMAGE_SIZE,
            class_mode="categorical",
            batch_size=BATCH_SIZE,
        )
        print('Created set for teaching\n')
        test_generator = test_datagen.flow_from_dataframe(
            test_df,
            DATASET_LOCATION,
            x_col=columns[0],
            y_col=columns[1],
            target_size=IMAGE_SIZE,
            class_mode="categorical",
            batch_size=BATCH_SIZE,
        )
        print('Created set for validation\n')
        # NOW WE TRAIN THE MODEL
        history = model.fit_generator(
            train_generator,
            epochs=EPOCHS,
            validation_data=test_generator,
            validation_steps=test_df.shape[0] // BATCH_SIZE,
            steps_per_epoch=train_df.shape[0] // BATCH_SIZE,
            verbose=1,
        )
        print('Trained frozen model. Now unfroze some\n')
        set_trainable = False
        for layer in conv_base.layers:
            if layer.name == 'block5_conv1':
                set_trainable = True
            if set_trainable:
                layer.trainable = True
            else:
                layer.trainable = False

        model.compile(
            loss=keras.losses.categorical_crossentropy,
            optimizer=keras.optimizers.RMSprop(lr=1e-5),
            metrics=["accuracy"],
        )

        history = model.fit_generator(
            train_generator,
            epochs=EPOCHS,
            validation_data=test_generator,
            validation_steps=test_df.shape[0] // BATCH_SIZE,
            steps_per_epoch=train_df.shape[0] // BATCH_SIZE,
            verbose=1,
        )

        print('Trained model\n')
        # save model and architecture to single file
        model.save(savename)
        print("Saved model to disk\n")
        return history
    except Exception as e:
        print(e)


def testModel(test_df, test_dir, name, columns):
    IMAGE_SIZE = (128, 128)
    INPUT_SHAPE = (128, 128, 3)
    # load model
    model = load_model(name)
    print('Loaded the model\n')
    # summarize model.
    model.summary()
    # test generator
    batches = 1
    sample_test_generator = ImageDataGenerator(rescale=1.0 / 255).flow_from_dataframe(
        test_df,
        test_dir,
        x_col=columns[0],
        y_col=columns[1],
        target_size=IMAGE_SIZE,
        class_mode="categorical",
        batch_size=batches,
    )
    label_map = (sample_test_generator.class_indices)
    print('Created test generator\n')
    # predict
    filenames = sample_test_generator.filenames
    nb_samples = len(filenames)
    # print('Len of filenames is:' + nb_samples)
    predict = model.predict_generator(sample_test_generator, np.ceil(nb_samples / batches))
    return predict, label_map
    # score = model.evaluate_generator(sample_test_generator)
    # print("Accuracy = ", score[1]*100, '%')


# check the indices and their categories on original dataset
def check_orig_labels(learnPd, learnImDir, train_cols):
    train_generator = ImageDataGenerator(rescale=1.0 / 255).flow_from_dataframe(
        learnPd,
        learnImDir,
        x_col=train_cols[0],
        y_col=train_cols[1],
        target_size=(128, 128, 3),
        class_mode="categorical",
    )
    print(len(train_generator.class_indices))
    return train_generator.class_indices


def change_keys_from_org_to_new(label_map_orig, label_map_test):
    res = {}
    for key, val in label_map_orig.items():
        # label_map_test[key] = label_map_orig[key]
        res[val] = key
    print(label_map_orig, label_map_test)
    return res
