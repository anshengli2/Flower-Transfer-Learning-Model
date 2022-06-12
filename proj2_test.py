###########################################################################
#               README
#  Ansheng Li
#  CS6384
#
#  TensorFlow -v 2.8.0
#  Numpy      -v 1.22.2
#  Pandas     -v 1.4.2
#
#  1. All files should be in the same folder
#     including: flowers_test, flowers_test.csv
#     my_model.h5, proj2_test.py
#  2. The code provided by the Professor runs
#     in the same fashion, by providing
#     --model --test.csv
#  3. Running the snippet
#       python proj2_test.py --model my_model.h5 --test_csv flowers_test.csv
#     Alternatively, the model names and test_csv name is default already
###########################################################################

import pandas as pd
import argparse
import tensorflow as tf
import os
import cv2
import numpy as np
import sklearn.preprocessing as preprocessing
# Note that you can save models in different formats. Some format needs to save/load model and weight separately.
# Some saves the whole thing together. So, for your set up you might need to save and load differently.


def load_model_weights(model, weights=None):
    my_model = tf.keras.models.load_model(model)
    my_model.summary()
    return my_model


def get_images_labels(df, img_height, img_width):
    test_images = []
    test_labels = []
    # Write the code as needed for your code
    for index, row in df.iterrows():
        test_labels.append(row[1])
        img = tf.io.read_file(row[0])
        img = decode_img(img, img_height, img_width)
        test_images.append(img)
    return np.array(test_images), np.array(test_labels)


def decode_img(img, img_height, img_width):
    # Convert the compressed string to a 3D uint8 tensor
    img = tf.io.decode_jpeg(img, channels=3)
    # Resize the image to the desired size
    return tf.image.resize(img, [img_height, img_width])


if __name__ == "__main__":
    print(pd.__version__)
    parser = argparse.ArgumentParser(description="Trasnfer Learning Test")
    parser.add_argument('--model', type=str,
                        default='my_model.h5', help='Saved model')
    parser.add_argument('--weights', type=str, default=None,
                        help='weight file if needed')
    parser.add_argument('--test_csv', type=str,
                        default='flowers_test.csv', help='CSV file with true labels')

    args = parser.parse_args()
    model = args.model
    weights = args.weights
    test_csv = args.test_csv

    test_df = pd.read_csv(test_csv)
    classes = ['astilbe', 'bellflower', 'black-eyed susan', 'calendula', 'california poppy',
               'carnation', 'common daisy', 'coreopsis', 'dandelion', 'iris', 'rose', 'sunflower', 'tulip']
    classes_enc = {'astilbe': 0, 'bellflower': 1, 'black-eyed susan': 2, 'calendula': 3, 'california poppy': 4,
                   'carnation': 5, 'common daisy': 6, 'coreopsis': 7, 'dandelion': 8, 'iris': 9, 'rose': 10, 'sunflower': 11, 'tulip': 12}

    # Rewrite the code to match with your setup
    img_size = 224
    test_images, test_labels = get_images_labels(test_df, img_size, img_size)
    # single_label = test_labels

    # onehot encoding for the 13 classes
    targets = np.array(classes)
    labelEnc = preprocessing.LabelEncoder()
    new_target = labelEnc.fit_transform(targets)
    onehotEnc = preprocessing.OneHotEncoder()
    onehotEnc.fit(new_target.reshape(-1, 1))
    targets_trans = onehotEnc.transform(new_target.reshape(-1, 1))
    labels_enc = targets_trans.toarray()

    labels = test_labels
    test_labels = []

    for flower in labels:
        flower = flower.strip()
        if flower in classes_enc:
            test_labels.append(labels_enc[classes_enc[flower]])
    test_labels = np.array(test_labels)

    my_model = load_model_weights(model)

    loss, acc = my_model.evaluate(test_images, test_labels, verbose=2)
    print('Test model, accuracy: {:5.5f}%'.format(100 * acc))

    # for i in range(0, len(test_images)):
    #     prediction_scores = my_model.predict(
    #         np.expand_dims(test_images[i], axis=0))
    #     predicted_index = np.argmax(prediction_scores)
    #     print(" Predicted label: " +
    #           classes[predicted_index] + "    Actual: " + single_label[i])
