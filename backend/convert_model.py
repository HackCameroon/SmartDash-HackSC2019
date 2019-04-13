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


# # A model class instance (class not shown)
# model = MyModelClass()
#
# # Load the weights from a file (.pth usually)
# state_dict = torch.load('data/')
#
# # Load the weights now into a model net architecture defined by our class
# model.load_state_dict(state_dict)
#
# # Create the right input shape (e.g. for an image)
# dummy_input = torch.randn(sample_batch_size, channel, height, width)
#
# torch.onnx.export(model, dummy_input, "onnx_model_name.onnx")

def create_ImageBunch(path):
    data = ImageDataBunch.from_folder(path, train=".", valid_pct=0.2, \
            ds_tfms=get_transforms(), size=281, num_workers=0, bs=32).normalize(imagenet_stats)
    return data

def load_model(data, model_file):
    learn = cnn_learner(data, models.resnet34, metrics=error_rate)
    learn.load(model_file)
    return learn

def main():
    path = Path('./data')
    data = create_ImageBunch(path)
    learn = load_model(data, 'stage-1')
    # Model = torch.save(learn.model, 'Weights.h5')
    model = learn.model
    state_dict = torch.load('./data/models/stage-1.pth')
    # model.load_state_dict(state_dict)
    dummy_input = Variable(torch.randn(1, 3, 281, 281))
    torch.onnx.export(learn.model, dummy_input, 'model.onnx', input_names=['image'], output_names=['accidentornot'], verbose=True)

    model_name = 'model.onnx'
    onnx_model = onnx.load(model_name)

    # onnx model --> Apple Core ML
    mlmodel = convert(onnx.load(model_name), image_input_names = ['image'], \
                      mode='classifier', class_labels="labels.txt")
    mlmodel.author = 'adithya'
    mlmodel.license = 'MIT'
    mlmodel.short_description = 'This model takes a dashcam picture and determines if it sees a crash'
    mlmodel.input_description['image'] = 'Dashcam Image'
    mlmodel.output_description['accidentornot'] = 'Confidence and label of accident'
    mlmodel.output_description['classLabel'] = 'Label of predicted accident or not'
    mlmodel.save(f'{model_name}.mlmodel')



if __name__ == '__main__':
    np.random.seed(42)
    main()
