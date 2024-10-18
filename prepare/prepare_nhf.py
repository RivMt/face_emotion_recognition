from PIL import Image
import os

# Convert
# https://www.kaggle.com/datasets/sudarshanvaidya/random-images-for-face-emotion-recognition

outer_names = ['test','train']
inner_names = ['angry', 'disgusted', 'fearful', 'happy', 'sad', 'surprised', 'neutral']

image_width = 48
image_height = 48
test_rate = 0.8

convert_dict = {
    "anger": "angry",
    "disgust": "disgusted",
    "fear": "fearful",
    "happiness": "happy",
    "sadness": "sad",
    "neutrality": "neutral",
    "surprise": "surprised",
}

# Resize images
os.makedirs(os.path.join('./data/', "test"), exist_ok=True)
os.makedirs(os.path.join('./data/', "train"), exist_ok=True)
emotions = convert_dict.keys()
for emotion in emotions:
    if emotion not in convert_dict:
        continue
    print(f"Resize images about {emotion}")
    os.makedirs(os.path.join('./data/test/', convert_dict[emotion]), exist_ok=True)
    os.makedirs(os.path.join('./data/train/', convert_dict[emotion]), exist_ok=True)
    files = os.listdir(f"./raw/nhf/{emotion}")
    for i in range(len(files)):
        file = files[i]
        typ = "train" if i < len(files) * test_rate else "test"
        image = Image.open(f"./raw/nhf/{emotion}/{file}")
        image.thumbnail((image_width, image_height), Image.Resampling.LANCZOS)
        image.save(f"./data/{typ}/{convert_dict[emotion]}/im-nhf{i}.png", "png")