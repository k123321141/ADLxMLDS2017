import os
from keras.models import *
import numpy as np
from scipy.misc import imresize, imsave
def main():

    zsamples = np.random.normal(size=(10 * 10, 100))
    generator = load_model('./generator_3.h5')
    out = generator.predict(zsamples)#.transpose((0, 2, 3, 1))
    print out.shape
    for i in range(100):
        print 'save'
        imsave('./early/%d.png' % i,out[i,:,:,:])


if __name__ == "__main__":
    main()
