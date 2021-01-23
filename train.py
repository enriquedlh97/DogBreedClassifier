import numpy as np
import os
# os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow_hub as hub
from dataset import get_loaders
from tqdm import tqdm

URL = "https://tfhub.dev/tensorflow/efficientnet/b0/feature-vector/1"
IMG_SIZE = 224  # Depended on the model. If efficientnet model is changed then the size is different
BATCH_SIZE = 32
NUM_CLASSES = 127
NUM_EPOCHS = 3
DATA_DIR = "data/"
MODEL_PATH = "efficientb0/"
LOAD_MODEL = False

os.environ["TFHUB_DOWNLOAD_PROGRESS"] = "1"
os.environ["TFHUB_CHACHE_DIR"] = "G:/My Drive/Tec/Semestre 9.5 - Invierno/Evaluacion y Admin de proyectos/Proyecto/MascotitasApp/loaded_model"


def get_model(url, img_size, num_classes):
    model = tf.keras.Sequential([
        hub.KerasLayer(url, trainable=True),
        layers.Dense(1000, activation="relu"),
        layers.Dropout(0.3),
        layers.Dense(num_classes, activation="softmax")
    ])
    model.build([None, img_size, img_size, 3])
    return model


@tf.function
def train_step(data, labels, acc_metric, model, loss_fn, optimizer):
    with tf.GradientTape() as tape:
        predictions = model(data, training=True)
        loss = loss_fn(labels, predictions)

    gradients = tape.gradient(loss, model. trainable_weights)
    optimizer.apply_gradients(zip(gradients, model.trainable_weights))
    acc_metric.update_state(labels, predictions)


def evaluate_model(ds_validation, model):
    accuracy_metric = keras.metrocs.SparseCategoricalAccuracy()
    for idx, (data, labels) in enumerate(ds_validation):
        data = data.permute(0, 2, 3, 1)
        data = tf.convert_to_tensor(np.array(data))
        labels = tf.convert_to_tensor(np.array(labels))
        y_pred = model(data, training=False)
        accuracy_metric.update_state(labels, y_pred)

    accuracy = accuracy_metric.result()
    print(f"Accuracy over validation set: {accuracy}")


def main():
    train_loader, dev_laoder = get_loaders(DATA_DIR+"train", DATA_DIR+"dev",
                                           BATCH_SIZE, IMG_SIZE)

    if LOAD_MODEL:
        print("Loading model")
        model = keras.models.load_model(MODEL_PATH)

    else:
        print("Building model")
        model = get_model(URL, IMG_SIZE, NUM_CLASSES)

    optimizer = keras.optimizers.Adam(lr=3e-4)
    loss_fn = keras.losses.SparseCategoricalCrossEntropy(from_logits=False)
    acc_metric = keras.metrics.SparseCategoricalAccuracy()

    # Training loop
    for epoch in range(NUM_EPOCHS):
        for idx, (data, labels) in enumerate(tqdm(train_loader)):
            # Swap access for num channels because we are loading data with pytorch
            # Pytorch: N x C x H x W
            # Tensorflow: N x H x W x C
            data = data.permute(0, 2, 3, 1)
            data = tf.convert_to_tensor(np.array(data))
            labels = tf.convert_to_tensor(np.array(labels))
            train_step(data, labels, acc_metric, model, loss_fn, optimizer)

            if idx % 150 == 0 and idx > 0:
                train_acc = acc_metric.result()
                print(f"Accuracy over epoch: {train_acc}")

                evaluate_model(dev_laoder, model)
                model.save(MODEL_PATH)

if __name__ == "__main__":
    main()
