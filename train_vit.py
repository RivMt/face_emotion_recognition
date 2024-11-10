import os

os.environ["XLA_FLAGS"]="--xla_gpu_cuda_data_dir=\"C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.3\""
os.environ["KERAS_BACKEND"] = "tensorflow"  # @param ["tensorflow", "jax", "torch"]
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import keras
from keras.src.legacy.preprocessing.image import ImageDataGenerator
from keras import layers
from keras import ops

import numpy as np
import matplotlib.pyplot as plt

from model_vit import image_size, patch_size, get_model, Patches, channels


# https://keras.io/examples/vision/image_classification_with_vision_transformer/

batch_size = 256
num_epochs = 400  # For real training, use num_epochs=100. 10 is a test value


train_dir = 'prepare/data/train'
val_dir = 'prepare/data/test'

train_datagen = ImageDataGenerator(rescale=1./255)
val_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(image_size,image_size),
        batch_size=batch_size,
        color_mode="grayscale",
        class_mode='sparse')

validation_generator = val_datagen.flow_from_directory(
        val_dir,
        target_size=(image_size,image_size),
        batch_size=batch_size,
        color_mode="grayscale",
        class_mode='sparse')


num_batches = len(train_generator)
random_batch_index = np.random.randint(num_batches)
random_batch = train_generator[random_batch_index]
images, labels = random_batch
random_image_index = np.random.randint(len(images))
image = images[random_image_index]

resized_image = ops.image.resize(
    ops.convert_to_tensor([image]), size=(image_size, image_size)
)
patches = Patches(patch_size)(resized_image)
print(f"Image size: {image_size} X {image_size}")
print(f"Patch size: {patch_size} X {patch_size}")
print(f"Patches per image: {patches.shape[1]}")
print(f"Elements per patch: {patches.shape[-1]}")

n = int(np.sqrt(patches.shape[1]))
plt.figure(figsize=(4, 4))
for i, patch in enumerate(patches[0]):
    ax = plt.subplot(n, n, i + 1)
    patch_img = ops.reshape(patch, (patch_size, patch_size, channels))
    plt.imshow(ops.convert_to_numpy(patch_img).astype("uint8"))
    plt.axis("off")


def run_experiment(model):
    learning_rate = 0.001
    weight_decay = 0.0001
    optimizer = keras.optimizers.AdamW(
        learning_rate=learning_rate, weight_decay=weight_decay
    )

    model.compile(
        optimizer=optimizer,
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[
            keras.metrics.SparseCategoricalAccuracy(name="accuracy"),
            keras.metrics.SparseTopKCategoricalAccuracy(2, name="top-2-accuracy"),
        ],
    )

    checkpoint_filepath = "output/checkpoint.weights.h5"
    checkpoint_callback = keras.callbacks.ModelCheckpoint(
        checkpoint_filepath,
        monitor="accuracy",
        save_best_only=True,
        save_weights_only=True,
    )

    history = model.fit(
        train_generator,
        steps_per_epoch=len(train_generator),
        validation_data=validation_generator,
        validation_steps=len(validation_generator),
        batch_size=batch_size,
        epochs=num_epochs,
        callbacks=[checkpoint_callback],
    )

    model.load_weights(checkpoint_filepath)
    _, accuracy, top_5_accuracy = model.evaluate(validation_generator, steps=len(validation_generator))
    print(f"Test accuracy: {round(accuracy * 100, 2)}%")
    print(f"Test top 2 accuracy: {round(top_5_accuracy * 100, 2)}%")

    return history




def plot_history(item):
    plt.plot(history.history[item], label=item)
    plt.plot(history.history["val_" + item], label="val_" + item)
    plt.xlabel("Epochs")
    plt.ylabel(item)
    plt.title("Train and Validation {} Over Epochs".format(item), fontsize=14)
    plt.legend()
    plt.grid()
    plt.show()
    plt.savefig(f"output/vit-{item}-{num_epochs}.png")



if __name__ == "__main__":
    vit_classifier = get_model(train_generator)
    history = run_experiment(vit_classifier)
    plot_history("loss")
    plot_history("top-2-accuracy")