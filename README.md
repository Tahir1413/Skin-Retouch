# Skin-Retouch
import numpy as np
import cv2
from sklearn.cluster import KMeans
from collections import Counter
import imutils
import pprint
from matplotlib import pyplot as plt


def extractSkin(image):
    # Taking a copy of the image
    img = image.copy()
    # Converting from BGR Colours Space to HSV
    img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Defining HSV Threadholds
    lower_threshold = np.array([0, 48, 80], dtype=np.uint8)
    upper_threshold = np.array([20, 255, 255], dtype=np.uint8)

    # Single Channel mask,denoting presence of colours in the about threshold
    skinMask = cv2.inRange(img, lower_threshold, upper_threshold)

    # Cleaning up mask using Gaussian Filter
    skinMask = cv2.GaussianBlur(skinMask, (5,5), 0)

    # Extracting skin from the threshold mask
    skin = cv2.bitwise_and(img, img, mask=skinMask)

    # Return the Skin image
    return cv2.cvtColor(skin, cv2.COLOR_HSV2BGR)
    

##Pretty Print The function makes print out the color information in a readable manner


def prety_print_data(color_info):
    for x in color_info:
        print(pprint.pformat(x))
        print()

# Get Image with  image=cv2.imread("FILE_NAME")
image = cv2.imread("C:/Users/New User/Desktop/Apparel/New folder/original.jpg")


# Apply Skin Mask
skin = extractSkin(image)

##plt.subplot(3, 1, 3)
img = cv2.add(skin, image) 

plt.imshow(cv2.cvtColor(cv2.addWeighted(img, 0.5, image, 0.5, 0), cv2.COLOR_BGR2RGB))

##plt.tight_layout()
plt.show()

