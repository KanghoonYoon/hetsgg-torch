
from __future__ import print_function, absolute_import, division
import os, sys

sys.path.append( os.path.normpath( os.path.join( os.path.dirname( __file__ ) , '..' , 'helpers' ) ) )
from csHelpers import *

from cityscapesscripts.evaluation.instance import *
from cityscapesscripts.helpers.csHelpers import *
import cv2
from hetsgg.utils import cv2_util


def instances2dict_with_polygons(imageFileList, verbose=False):
    imgCount     = 0
    instanceDict = {}

    if not isinstance(imageFileList, list):
        imageFileList = [imageFileList]

    if verbose:
        print("Processing {} images...".format(len(imageFileList)))

    for imageFileName in imageFileList:
        img = Image.open(imageFileName)

        imgNp = np.array(img)

        instances = {}
        for label in labels:
            instances[label.name] = []

        for instanceId in np.unique(imgNp):
            if instanceId < 1000:
                continue
            instanceObj = Instance(imgNp, instanceId)
            instanceObj_dict = instanceObj.toDict()

            if id2label[instanceObj.labelID].hasInstances:
                mask = (imgNp == instanceId).astype(np.uint8)
                contour, hier = cv2_util.findContours(
                    mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

                polygons = [c.reshape(-1).tolist() for c in contour]
                instanceObj_dict['contours'] = polygons

            instances[id2label[instanceObj.labelID].name].append(instanceObj_dict)

        imgKey = os.path.abspath(imageFileName)
        instanceDict[imgKey] = instances
        imgCount += 1

        if verbose:
            print("\rImages Processed: {}".format(imgCount), end=' ')
            sys.stdout.flush()

    if verbose:
        print("")

    return instanceDict

def main(argv):
    fileList = []
    if (len(argv) > 2):
        for arg in argv:
            if ("png" in arg):
                fileList.append(arg)
    instances2dict_with_polygons(fileList, True)

if __name__ == "__main__":
    main(sys.argv[1:])
