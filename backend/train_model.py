from fastai.vision import *

def verify_dataset(path, classes):
	for c in classes:
		print(c)
		verify_images(path/c, delete=True, max_size=500)

def create_ImageBunch(path):
    data = ImageDataBunch.from_folder(path, train=".", valid_pct=0.2, \
            ds_tfms=get_transforms(), size=281, num_workers=0, bs=32).normalize(imagenet_stats)
    return data

def train_model(data):
    learn = cnn_learner(data, models.resnet34, metrics=error_rate)
    learn.fit_one_cycle(5)
    learn.save('stage-2')
    return learn

def load_model(data, model_file):
    learn = cnn_learner(data, models.resnet34, metrics=error_rate)
    learn.load(model_file)
    return learn

def main():
    path = Path('./data')
    verify_dataset(path, ['accident', 'no_accident'])
    data = create_ImageBunch(path)
    print(data.classes, data.c, len(data.train_ds), len(data.valid_ds))
    learn = train_model(data)
    return


if __name__ == '__main__':
    np.random.seed(42)
    main()
