import numpy as np
from PIL import Image


class CsvImgSet:

    def __init__(self):
        self.X = np.loadtxt('../Data/handwritten/handwritten_digits_images.csv', delimiter=',', dtype=np.uint8)
        self.y = np.loadtxt('../Data/handwritten/handwritten_digits_labels.csv', delimiter=',', dtype=np.uint8)

    def __getitem__(self, index):
        return self.X[index], self.y[index]

    def __len__(self):
        return self.X.shape[0]


def save_image(i, img, path, f):
    # print(type(img))
    img = Image.fromarray(img, 'L')
    img.save(path + str(i) + '/' + str(f) + '.png')


csvImgSet = CsvImgSet()
path = ["../Data/MNIST/train/", "../Data/MNIST/val/", "../Data/MNIST/test/"]
zero_to_nine = [[] for i in range(10)]
n_samples = len(csvImgSet)
for i in range(n_samples):
    zero_to_nine[csvImgSet[i][1]].append(csvImgSet[i][0].reshape(28, 28))

for i in range(10):
    n_samples = len(zero_to_nine[i])
    f = 0
    for j in range(n_samples):
        save_image(i, zero_to_nine[i][j], path[f], j)
        if j >= int(n_samples*0.85):
            break
        elif j >= int(n_samples*0.7):
            f = 1
