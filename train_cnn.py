import argparse
import os

from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator

import model_cnn as md
from plot import plot_model_history

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
train_dir = 'prepare/data/train'
val_dir = 'prepare/data/test'

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
        class_mode='categorical')

validation_generator = val_datagen.flow_from_directory(
        val_dir,
        target_size=(48,48),
        batch_size=batch_size,
        color_mode="grayscale",
        class_mode='categorical')

# Create the model
model = md.get_model()

# If you want to train the same model or try other models, go for this
model.compile(loss='categorical_crossentropy',optimizer=Adam(learning_rate=0.0001, decay=1e-6),metrics=['accuracy'])
model_info = model.fit(
        train_generator,
        steps_per_epoch=num_train // batch_size,
        epochs=num_epoch,
        validation_data=validation_generator,
        validation_steps=num_val // batch_size)
model.save_weights(f'{args.output}/model{batch_size}-{num_epoch}.weights.h5')
plot_model_history(model_info, display=not args.cli, filename=f"plot{batch_size}-{num_epoch}.png")