import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plot
plot.style.use('ggplot')

files = [file_i
            for file_i in os.listdir('faces')
            if file_i.endswith('.jpg')]

images = []
for f in files:
    img = plot.imread('faces/' + f)
    if img.shape != (150, 150, 3):
        os.remove('faces/' + f)
    else:
        images.append(img)

data = np.array(images)
print(data.shape)

mean = np.mean(data, axis=0)
plot.imsave('testing.jpg', mean.astype(np.uint8))
