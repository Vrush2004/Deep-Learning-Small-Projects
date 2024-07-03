#*****processing the image by using matplotlib library*****
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

#loading an image through matplotlib .image module
img = mpimg.imread("C:/Users/Dell/Desktop/Deep Learning Projects/Image Processing/dog.jpg")
type(img)
print(img.shape)
print(img)

#displaying the image from numpy array
img_plot = plt.imshow(img)
plt.show()

#*****Processing the image using Pillow library*****
from PIL import Image
img = Image.open('C:/Users/Dell/Desktop/Deep Learning Projects/Image Processing/dog.jpg')
img_resized = img.resize((200,200))
img_resized.save('dog_image_resized.jpg')

#displaying the image from numpy array
img_res = mpimg.imread("C:/Users/Dell/Desktop/Deep Learning Projects/Image Processing/dog_image_resized.jpg")
img_res_plot = plt.imshow(img_res)
plt.show()
print(img_res.shape)

#*****Convert RGB images to Crayscale image using OpenCV*****
import cv2 
img1 = cv2.imread("C:/Users/Dell/Desktop/Deep Learning Projects/Image Processing/dog.jpg")
type(img1)
grayscale_image = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)
print(grayscale_image.shape)

#displaying the image 
cv2.imshow('Image Window',grayscale_image)

#saving the grayscale image
cv2.imwrite('dog_grayscale_image.jpg')


