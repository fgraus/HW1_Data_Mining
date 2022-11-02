import numpy as np
import constants
from getData import getDataFromFile
from computeSimilarity import computeAccuracy


def mainCrossValidation():

    allReviews = getDataFromFile()

    generalAccuracy = 0

    reviewsChuncks = np.array(np.array_split(allReviews, constants.numberOfChuncks), dtype=object)

    for i in range(len(reviewsChuncks)):
        test = reviewsChuncks[i]
        train = np.concatenate(np.append(reviewsChuncks[:i], reviewsChuncks[i+1:], axis=0))

        generalAccuracy += computeAccuracy(train, test)

    mediumAccuracy = generalAccuracy / constants.numberOfChuncks
    return mediumAccuracy

#mainCrossValidation()