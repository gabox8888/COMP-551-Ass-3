import numpy
import scipy.misc

x = numpy.fromfile('data/train_x.bin', dtype='uint8')
x = x.reshape((100000,60,60))
scipy.misc.imsave('./data/image1.png',x[1]) # to visualize only
# scipy.misc.imshow(x[0]) # to visualize only

y = numpy.loadtxt(open("data/train_y.csv","rb"),delimiter=",",skiprows=1)
print(y)