import os
import cv2
from img2vec_pytorch import Img2Vec
from PIL import Image
from torchvision.datasets.folder import pil_loader

# prepare the data

img2vec = Img2Vec()

data_dir = './data/weather_dataset'
train_dir = os.path.join(data_dir, 'train')
val_dir = os.path.join(data_dir, 'val')

data={}
for j,dir_ in enumerate([train_dir, val_dir]):
    features = []
    labels = []
    for category in os.listdir(dir_):
        for img_path in os.listdir(os.path.join(dir_, category)):
            img_path_ = os.path.join(dir_, category, img_path)
            img = pil_loader(img_path_)

            img_features = img2vec.get_vec(img)

            features.append(img_features)
            labels.append(category)
    data[['training_data', 'validation_data'][j]] = features
    data[['training_labels', 'validation_labels'][j]] = labels

print(data.keys())

# train model

# test performance

# save the model