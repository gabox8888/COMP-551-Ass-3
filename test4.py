import scipy.ndimage as ndimage

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm

import scipy
import numpy as np

def load_dataset():
    x = np.fromfile('data/trainBW.bin', dtype='uint8')
    x = x.reshape((-1,1,60,60))
    x = x/ np.float32(256)
    return x

def apply_threshold(img,blur):
    for i in range(len(blur)):
        for j in range(len(blur[0])):
            if blur[i][j] < 0.9:
                img[i][j] = 0.0
    return img

def find_centers(img, original):
    current_avg = 0
    subJ = 0
    subK = 0
    image = img[0:28,0:28]
    for j in range(40):
        for k in range(40):
            new_img = img[j:j+20,k:k+20]
            if new_img.mean() > current_avg :
                subJ = j
                subK = k
                stuff = np.zeros(shape=(28,28))
                stuff[4:24,4:24] = original[j:j+20,k:k+20]
                current_avg = new_img.mean()
                test_img =stuff
                image = test_img
    temp_img = np.copy(img)
    temp_img[subJ:subJ+20,subK:subK+20] = np.zeros((20,20),dtype=int)
    return image, temp_img

def try_something(img):
    for i in range(len(img)):
        for j in range(len(img[0])):
            if img[i][j] < 0.2:
                img[i][j] = 1.0
            else:
                img[i][j] = 0.0
    return img

def remove_background(img,chop):
    width = len(img)
    height = len(img[0])
    for y in range(height):
        for x in range(width):
            # Make sure we're on a dark pixel.
            if img[x, y] > 0.9:
                continue
            # Keep a total of non-white contiguous pixels.
            total = 0
            # Check a sequence ranging from x to image.width.
            for c in range(x, width):
                # If the pixel is dark, add it to the total.
                if img[c, y] > 0.9:
                    total += 1
                # If the pixel is light, stop the sequence.
                else:
                    break
                # If the total is less than the chop, replace everything with white.
                if total <= chop:
                    for c in range(total):
                        img[x + c, y] = 0.0
                # Skip this sequence we just altered.
                x += total
    # Iterate through the columns.
    for x in range(width):
        for y in range(height):
            # Make sure we're on a dark pixel.
            if img[x, y] > 0.9:
                continue
            # Keep a total of non-white contiguous pixels.
            total = 0
            # Check a sequence ranging from y to image.height.
            for c in range(y, height):
                # If the pixel is dark, add it to the total.
                if img[x, c] > 0.9:
                    total += 1
                # If the pixel is light, stop the sequence.
                else:
                    break
            # If the total is less than the chop, replace everything with white.
            if total <= chop:
                for c in range(total):
                    img[x, y + c] = 0.0
            # Skip this sequence we just altered.
            y += total
    return img
    


def main():
    x = load_dataset()
    img = x[18089][0]
    plt.imshow(img, cmap=cm.binary)
    plt.show()

    # img = remove_background(img,2)
    # plt.imshow(img, cmap=cm.binary)
    # plt.show()


    # plt.imshow(try_something(sob), cmap=cm.binary)
    # plt.show()

    # median = ndimage.median_filter(img, 4)
    # plt.imshow(median, cmap=cm.binary)
    # plt.show()

    # blur = ndimage.gaussian_filter(img, sigma=1, order=0)
    # plt.imshow(blur, cmap=cm.binary)
    # plt.show()

    threshold = apply_threshold(img,img)
    plt.imshow(img, cmap=cm.binary)
    plt.show()

    # sx = ndimage.sobel(threshold, axis=0, mode='constant')
    # sy = ndimage.sobel(threshold, axis=1, mode='constant')
    # sob = np.hypot(sx, sy)
    # plt.imshow(sob, cmap=cm.binary)
    # plt.show()

    image1,i = find_centers(threshold,img)
    image2,j = find_centers(i,img)
    # image3,k = find_centers(j,img)
    plt.imshow(image1, cmap=cm.binary)
    plt.show()
    plt.imshow(image2, cmap=cm.binary)
    plt.show()
    # plt.imshow(image3[0], cmap=cm.binary)
    # plt.show()

if __name__ == '__main__':main()
