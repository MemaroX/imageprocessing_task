import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the image
img = cv2.imread('test.jpg')

# Convert the image from BGR to RGB
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Apply histogram to each color channel
color = ('r', 'g', 'b')
for i, col in enumerate(color):
    histr = cv2.calcHist([img], [i], None, [256], [0, 256])
    plt.plot(histr, color=col)
    plt.xlim([0, 256])
plt.title('Histogram for original image')
plt.show()

# Increase contrast
img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
img_yuv[:,:,0] = cv2.equalizeHist(img_yuv[:,:,0])
img_output = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)

# Reduce brightness
brightness_matrix = np.ones(img.shape, dtype='uint8') * 60
img_bright = cv2.subtract(img_output, brightness_matrix)

# Apply histogram to each color channel for the processed image
for i, col in enumerate(color):
    histr = cv2.calcHist([img_bright], [i], None, [256], [0, 256])
    plt.plot(histr, color=col)
    plt.xlim([0, 256])
plt.title('Histogram for processed image')
plt.show()
