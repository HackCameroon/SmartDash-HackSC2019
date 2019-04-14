from fastai.vision import *

def create_ImageBunch(path):
    data = ImageDataBunch.from_folder(path, train=".", valid_pct=0.2, \
            ds_tfms=get_transforms(), size=281, num_workers=0, bs=32).normalize(imagenet_stats)
    return data

def load_model(data, model_file):
    learn = cnn_learner(data, models.resnet34, metrics=error_rate)
    learn.load(model_file)
    return learn

def test_on_image(learn, img_path):
    trn_tfms, val_tfms = tfms_from_model(learn, (281,500))
    im = trn_tfms(Image.open(img_path))
    preds = predict_array(im[None])
    print(preds)
    return

def main():
    path = Path('./data')
    data = create_ImageBunch(path)
    learn = load_model(data, 'stage-3')
    learn.sched.plot()
    test_on_image(learn, './data/accident/frame0001.png')
