import torch
import torch.onnx
import torchvision
import torch.nn as nn
import numpy as np
from torch.autograd import Variable

from fastai.vision import *
from fastai.imports import *

import onnx
from onnx_coreml import convert

def verify_dataset(path, classes):
	for c in classes:
		print(c)
		verify_images(path/c, delete=True, max_size=500)

def create_ImageBunch(path):
    data = ImageDataBunch.from_folder(path, train=".", valid_pct=0.2, \
            ds_tfms=get_transforms(), size=(281,500), num_workers=0, bs=24).normalize(imagenet_stats)
    return data

def train_model(data):
	learn = cnn_learner(data, models.resnet34, metrics=error_rate)
	learn.fit_one_cycle(4)
	learn.save('stage-5')
	return learn

def test_on_image(learn, img_path):
    trn_tfms, val_tfms = tfms_from_model(learn, (281,500))
    im = trn_tfms(Image.open(img_path))
    preds = predict_array(im[None])
    print(preds)
    return

def main():
	path = Path('./data')
	verify_dataset(path, ['accident', 'no_accident'])
	data = create_ImageBunch(path)
	print(data.classes, data.c, len(data.train_ds), len(data.valid_ds), data.valid_ds)
	learn = train_model(data)
	print(learn.get_preds())

	dummy_input = Variable(torch.randn(1, 3, 281, 500))
	torch.onnx.export(learn.model, dummy_input, 'model-standard.onnx', input_names=['image'], output_names=['prob'], verbose=True)
	model_name = 'model-standard.onnx'
	onnx_model = onnx.load(model_name)
	mlmodel = convert(onnx.load(model_name), image_input_names = ['image'], mode='classifier', class_labels="labels.txt")
	mlmodel.author = 'adithya'
	mlmodel.license = 'MIT'
	mlmodel.short_description = 'This model takes a dashcam picture and determines if it sees a crash!'
	mlmodel.input_description['image'] = 'Dashcam Image'
	mlmodel.output_description['prob'] = 'Confidence and label of accident'
	mlmodel.output_description['classLabel'] = 'Label of predicted accident or not'
	mlmodel.save(f'AccidentDetection.mlmodel')

	# test_on_image(learn, './data/accident/frame0001.png')

	return

if __name__ == '__main__':
    np.random.seed(42)
    main()
