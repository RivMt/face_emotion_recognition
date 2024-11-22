import argparse
import os

from sklearn.utils import class_weight
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator

import model as md

# Init arguments
parser = argparse.ArgumentParser()
parser.add_argument('--test', action='store_true', default=False)
parser.add_argument('--cli', action='store_true', default=False)
parser.add_argument('--batch', action='store', default=64, type=int)
parser.add_argument('--epoch', action='store', default=50, type=int)
parser.add_argument('--output', action='store', default='./output', type=str)
args = parser.parse_args()
test_mode = args.test

# Set log level
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' if args.cli else '2'

# Define data generators
train_dir = 'prepare/data/bin-train'
val_dir = 'prepare/data/bin-test'

num_train = sum([len(os.listdir(os.path.join(train_dir, d))) for d in os.listdir(train_dir)])
num_val = sum([len(os.listdir(os.path.join(val_dir, d))) for d in os.listdir(val_dir)])
batch_size = args.batch
num_epoch = 2 if test_mode else args.epoch
print(f"Initialize learning: Train={num_train}, Val={num_val}")

train_datagen = ImageDataGenerator(rescale=1./255)
val_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(48,48),
    batch_size=batch_size,
    color_mode="grayscale",
    class_mode="binary",
    follow_links=True,
)
print(train_generator.class_indices)
print(train_generator.classes)
print(f"Class 0 samples: {sum(train_generator.classes == 0)}")
print(f"Class 1 samples: {sum(train_generator.classes == 1)}")

validation_generator = val_datagen.flow_from_directory(
    val_dir,
    target_size=(48,48),
    batch_size=batch_size,
    color_mode="grayscale",
    class_mode="binary",
    follow_links=True,
)


# plots accuracy and loss curves
def plot_model_history(model_history):
    """
    Plot Accuracy and Loss curves given the model_history
    """
    fig, axs = plt.subplots(1,2,figsize=(15,5))
    # summarize history for accuracy
    axs[0].plot(range(1,len(model_history.history['accuracy'])+1),model_history.history['accuracy'])
    axs[0].plot(range(1,len(model_history.history['val_accuracy'])+1),model_history.history['val_accuracy'])
    axs[0].set_title('Model Accuracy')
    axs[0].set_ylabel('Accuracy')
    axs[0].set_xlabel('Epoch')
    axs[0].set_ylim(0, 1)
    axs[0].set_xticks(np.arange(1,len(model_history.history['accuracy'])+1))
    axs[0].legend(['train', 'val'], loc='best')
    # summarize history for loss
    axs[1].plot(range(1,len(model_history.history['loss'])+1),model_history.history['loss'])
    axs[1].plot(range(1,len(model_history.history['val_loss'])+1),model_history.history['val_loss'])
    axs[1].set_title('Model Loss')
    axs[1].set_ylabel('Loss')
    axs[1].set_xlabel('Epoch')
    axs[1].set_xticks(np.arange(1,len(model_history.history['loss'])+1))
    axs[1].legend(['train', 'val'], loc='best')
    fig.savefig(f'{args.output}/bin/plot{batch_size}-{num_epoch}.png')
    if not args.cli:
        plt.show()

# Create the model
model = md.getModel()

# If you want to train the same model or try other models, go for this
model.compile(
    loss='binary_crossentropy',
    optimizer=Adam(learning_rate=0.0001, decay=1e-6),
    metrics=['accuracy'],
)
class_weights = class_weight.compute_class_weight(
    class_weight='balanced',
    classes=np.unique(train_generator.classes),
    y=train_generator.classes
)
class_weights = dict(enumerate(class_weights))
model_info = model.fit(
    train_generator,
    steps_per_epoch=num_train // batch_size,
    epochs=num_epoch,
    validation_data=validation_generator,
    validation_steps=num_val // batch_size,
    class_weight=class_weights,
)
model.save_weights(f'{args.output}/bin/model{batch_size}-{num_epoch}.weights.h5')
plot_model_history(model_info)