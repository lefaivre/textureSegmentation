# ----------------------------------------------------------
# Adam Lefaivre (001145679)
# Cpsc 5990
# Final Program Project
# Dr. Howard Cheng
# ----------------------------------------------------------

import cv2
import math
import numpy as np
import _utils
from scipy import signal
import argparse
from scipy import ndimage
import os

# This is used by the brute force convolution of the moment mask
def centralPixelMomentCalculation(img, row, col, W, p, q):

    m_pq = 0.0
    height, width = img.shape
    windowHeight, windowWidth, initm, initn = _utils.getRanges_for_window_with_adjust(row, col, height, width, W)
    for m in range(windowHeight + 1):
        for n in range(windowWidth + 1):
            truem = initm + m
            truen = initn + n
            x_m = (truem - row) / math.floor(W / 2)
            y_n = (truen - col) / math.floor(W / 2)
            x_m_p = pow((x_m), p)
            y_n_p = pow((y_n), q)
            m_pq += (img[truem][truen] * x_m_p * y_n_p)

    return m_pq

# The brute force spatial convolution of the transducer is also implemented for completness
def centralPixelTangentCalculation_bruteForce(img, row, col, sigma, L, M):

    height, width = img.shape
    windowHeight, windowWidth, inita, initb = _utils.getRanges_for_window_with_adjust(row, col, height, width, L)

    sum = 0.0
    for a in range(windowHeight + 1):
        for b in range(windowWidth + 1):
            truea = inita + a
            trueb = initb + b
            sum += math.fabs(math.tanh(sigma * (img[truea][trueb] - M)))
    return sum / pow(L, 2)

# This convolves the input image with the moments mask.  The user can choose to use this function.
# Moment output is calculated on the fly
def computeMomentImage_bruteForce(img, W, p, q):

    height, width = img.shape
    copy = np.zeros(img.shape)
    for row in range(height):
        for col in range(width):
            copy[row][col] = centralPixelMomentCalculation(img, row, col, W, p, q)

    #copy = cv2.normalize(copy, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    # Normalization sets the input to the active range [-2,2] this becomes [-200,200] with sigma
    copy = cv2.normalize(copy, alpha=-200, beta=200, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

    return copy

# Instead of convolving the moments mask spatially, the user can choose to call this function instead
# It is the default function for the moment mask convolution.
# Moment masks are made beforehand, and then convolved.
def computeMomentImageOptimized(img, W, p, q):
    x_arr = np.empty((W, W))
    y_arr = np.empty((W, W))
    x_arr.fill(0)
    y_arr.fill(0)

    for i in range(0, W):
        for j in range(0, W):
            x_arr[i][j] -= math.floor(W / 2)
            x_arr[i][j] += i

    x_mp = np.power(x_arr, p)

    for i in range(0, W):
        for j in range(0, W):
            y_arr[i][j] -= math.floor(W / 2)
            y_arr[i][j] += j

    y_nq = np.power(y_arr, q)
    kernel = np.multiply(y_nq, x_mp)
    out = signal.convolve2d(img, kernel, mode='same', boundary='fill')
    # out = ndimage.convolve(img, kernel, mode='constant', cval=0.0)
    # out = cv2.normalize(out, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    # Normalization sets the input to the active range [-2,2] this becomes [-200,200] with sigma
    out = cv2.normalize(out, alpha=-200, beta=200, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

    return out

# This generates all p,q combinations and then calls the convolution corresponding
# to those p,q values.
def getAllMomentImages(img, W, pqThreshold, bruteForceMomentCalc):
    pqlist = []
    momentImages = []

    # Acquire p + q combinations
    for q in range(pqThreshold + 1):
        for p in range(0, q + 1):
            if ((p + q) > pqThreshold):
                break
            else:
                pqlist.append([p, q])

    reverseList = []

    for pq in pqlist:
        p = pq[0]
        q = pq[1]
        if (p != q):
            reverseList.append(pq[::-1])
    pqlist.extend(reverseList)
    pqlist = sorted(pqlist)

    # Convole w.r.t. all p,q combinations
    for pqVals in pqlist:
        p = pqVals[0]
        q = pqVals[1]
        if (bruteForceMomentCalc):
            momentImage = computeMomentImage_bruteForce(img, W, p, q)
        else:
            momentImage = computeMomentImageOptimized(img, W, p, q)
        momentImages.append(momentImage)

    return momentImages

# The activation function, with regular smoothing.
def nonLinearTransducer(img, momentImages, L):
    sigma = 0.01
    featureImages = []
    for momentImage in momentImages:
        height, width = momentImage.shape
        copy = np.zeros(img.shape)
        M = momentImage.mean()
        for row in range(height):
            for col in range(width):
                # copy[row][col] = centralPixelTangentCalculation_bruteForce(momentImage, row, col, sigma, L, M)
                copy[row][col] = (sigma * (momentImage[row][col] - M))

        copy = cv2.blur(copy, (L, L), borderType=cv2.BORDER_REFLECT_101)
        featureImages.append(copy)

    return featureImages

# Our main driver function to return the segmentation of the input image.
def runMoments(args):

    infile = args.infile
    if (not os.path.isfile(infile)):
        print infile, ' is not a file!'
        exit(0)

    outfile = args.outfile
    printlocation = os.path.dirname(os.path.abspath(outfile))
    _utils.deleteExistingSubResults(printlocation)

    k_clusters = args.k
    greyOutput = args.c
    printIntermediateResults = args.i

    W = args.W
    if((W % 2) == 0):
        print 'size of moments window is not odd, using next odd number'
        W += 1

    L_transducerWindowSize = args.L
    if ((L_transducerWindowSize % 2) == 0):
        print 'size of moments window is not odd, using next odd number'
        L_transducerWindowSize += 1

    pqThreshold = args.pq
    spatialWeight = args.spw
    bruteForceMomentCalc = args.b
    img = cv2.imread(infile, cv2.CV_LOAD_IMAGE_GRAYSCALE)

    print "Applying moment masks, If brute force was selected please wait..."
    momentImages = getAllMomentImages(img, W, pqThreshold, bruteForceMomentCalc)

    if(printIntermediateResults):
        _utils.printFeatureImages(momentImages, "filter", printlocation)

    print "Applying nonlinear transduction with averaging"
    featureImages = nonLinearTransducer(img, momentImages, L_transducerWindowSize)

    if(printIntermediateResults):
        _utils.printFeatureImages(featureImages, "feature", printlocation)

    featureVectors = _utils.constructFeatureVectors(featureImages, img)
    featureVectors = _utils.normalizeData(featureVectors, True, spatialWeight=spatialWeight)

    print "Clustering..."
    labels = _utils.clusterFeatureVectors(featureVectors, k_clusters)
    _utils.printClassifiedImage(labels, k_clusters, img, outfile, greyOutput)

# For running the program on the command line
def main():

    # initialize
    parser = argparse.ArgumentParser()

    # Required arguments
    parser.add_argument("-infile", required=True)
    parser.add_argument("-outfile", required=True)

    parser.add_argument('-k', help='Number of clusters', type=_utils.check_positive_int, required=True)
    parser.add_argument('-W', help='Size of the moments window', type=_utils.check_positive_int, required=True)
    parser.add_argument('-L', help='Size of the smoothing window', type=_utils.check_positive_int, required=True)

    # Optional arguments
    parser.add_argument('-spw', help='Spatial weight of the row and columns for clustering, DEFAULT = 1', nargs='?', const=1,
                        type=_utils.check_positive_float, default=1, required=False)
    parser.add_argument('-pq', help='the pq threshold, DEFAULT = 2', nargs='?', const=2, default=2,
                        type=_utils.check_positive_int, required=False)
    parser.add_argument('-b', help='Brute force moment convolution? True/False, DEFAULT = False', nargs='?', const=False,
                        default=False, type=bool, required=False)
    parser.add_argument('-c', help='Output grey? True/False, DEFAULT = False', nargs='?', const=False, default=False,
                        type=bool, required=False)
    parser.add_argument('-i', help='Print intermediate results (filtered/feature images)? True/False, DEFAULT = False', nargs='?', const=False, default=False,
                        type=bool, required=False)

    args = parser.parse_args()
    runMoments(args)

if __name__ == "__main__":
    main()