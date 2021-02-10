
import sys
import cv2 as cv
import numpy as np
import matplotlib
from matplotlib.pyplot import imshow
import os, os.path

from utils import Utils

# Object to acquire and transform the input picture in a probability distribution
class ImageProcessing(object):

    def __init__(self, pts=[], path=None, kernel_size=None, crop_points=[], parameters={}):

        self.input="Fixed_TEST"

        self.pts = pts
        self.crop_points = crop_points

        if(path != None):
            self.path = path
            self.count = 0
            self.input="FOLDER"
            #Upload list of files
            self.files = [name for name in os.listdir(path) if (os.path.isfile(path+name) and ".png" in name)]
            self.files.sort(key=lambda x: int( (x.split(".")[-2]).split("_")[1] ))
            self.files = [self.path + f for f in self.files]

        if(kernel_size == None):
            self.kernel_size=7
        else:
            self.kernel_size=kernel_size
        return

    def create_mask(self, image, verbose=False, tag=""):#a, b=False):
        #img = cv.imread(cv.samples.findFile("star_night.jpg"))
        if image is None:
            sys.exit("Could not read the image.")

        # white color mask
        image = cv.cvtColor(image,cv.COLOR_BGR2HLS)
        lower = np.uint8([0, 200, 0])
        upper = np.uint8([255, 255, 255])
        white_mask = cv.inRange(image, lower, upper)
        # yellow color mask
        lower = np.uint8([10, 0,   100])
        upper = np.uint8([40, 255, 255])
        yellow_mask = cv.inRange(image, lower, upper)
        # combine the mask
        mask = cv.bitwise_or(white_mask, yellow_mask)

        #mask=white_mask
        if verbose:
            cv.imshow("Mask", mask)
            #cv.imwrite('../Mask'+str(self.count)+'_'+tag+'.png', mask)
            k = cv.waitKey(0)

        return mask

    def crop_image(self, image, x_top_left, y_top_left, x_bottom_right, y_bottom_right, verbose=False, tag=""):

        # Crop image from y_bottom_right, x_top_left to y_bottom_right, x_bottom_right

        crop = image[y_top_left:y_bottom_right, x_top_left:x_bottom_right]

        if verbose:
            cv.imshow("Original image", image)
            k = cv.waitKey(1)
            cv.imshow("Cropped image", crop)
            #cv.imwrite('../Cropped'+str(self.count)+'_'+tag+'.png', crop)
            k = cv.waitKey(0)

        return crop

    def blur(self, image, verbose=False, tag=""):#a, b=False):

        img = cv.GaussianBlur(image,(self.kernel_size*3,self.kernel_size),cv.BORDER_DEFAULT)

        if verbose:
            cv.imshow("Blur", img)
            #cv.imwrite('../Blur'+str(self.count)+'_'+tag+'.png', img)
            k = cv.waitKey(0)

        return img

    def grey_image(self, img, verbose=False, tag=""):
        img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

        if verbose:
            cv.imshow("Grey", img)
            #cv.imwrite('../Grey'+str(self.count)+'_'+tag+'.png', img)
            k = cv.waitKey(0)

        return img

    def dimensions(self, img, verbose=False):

        h, w = img.shape

        return [h, w]

    def IPM(self, image, verbose=False, tag=""):

        utils = Utils()

        # pts correspond to the four points of the image

        if(verbose):
            # loop over the points and draw them on the image
            for (x, y) in self.pts:
                cv.circle(image, (x, y), 5, (0, 255, 0), -1)

        # apply the four point tranform to obtain a "birds eye view"
        warped = utils.four_point_transform(image, self.pts)

        if(verbose):
            # show the original and warped images
            cv.imshow("Original", image)
            k = cv.waitKey(1)
            cv.imshow("Warped", warped)
            #cv.imwrite('../IPM'+str(self.count)+'_'+tag+'.png', warped)
            k = cv.waitKey(0)

        return warped

    def get_lines_pdf(self, verbose=False, img=None):

        if(img==None):
            img  = self.acquire_frame(False)


        img  = self.crop_image(img, self.crop_points[0], self.crop_points[1], self.crop_points[2], self.crop_points[3], False)

        if(len(self.pts)!=0):
            img  = self.IPM(img, False)

        img1 = self.crop_image(img, 0, 0, int(img.shape[1]/2), int(img.shape[0]), False, tag="l")
        img2 = self.crop_image(img, int(img.shape[1]/2), 0, int(img.shape[1]), int(img.shape[0]), False, tag="r")

        pdf  = self.blur(self.create_mask(self.bright_contr(img)))
        pdf1 = self.blur(self.create_mask(self.bright_contr(img1, verbose=False, tag="l"),False, tag="l"),False, tag="l")
        pdf2 = self.blur(self.create_mask(self.bright_contr(img2, verbose=False, tag="r"),False, tag="r"),False, tag="r")

        return pdf, pdf1, pdf2, img, img1, img2

    def get_raw_image(self, verbose=False, img=None, tag=""):
        img=self.acquire_frame()

        if(len(self.pts)!=0):
            img  = self.IPM(img, False)

        img  = self.grey_image(img)
        img1 = self.crop_image(img, 0, 0, int(img.shape[1]/2), int(img.shape[0]), False)
        img2 = self.crop_image(img, int(img.shape[1]/2 ), 0, int(img.shape[1]), int(img.shape[0]), False)

        return img1, img2

    def acquire_frame(self, verbose=False, tag=""):
        if(self.input=="FOLDER"):
            self.count += 1
            img = cv.imread(self.files[self.count])
        else:
            img = cv.imread("../TestStreetHome.png")

        if verbose:
            cv.imshow("Aquired frame", img)
            #cv.imwrite('../AquiredFrame'+str(self.count)+'_'+tag+'.png', img)
            k = cv.waitKey(0)

        return img

    def bright_contr(self, img, brightness = -60, contrast = 100, verbose=False, tag=""):#-60 100   -120 127
        img = np.int16(img)
        img = img * (contrast/127+1) - contrast + brightness
        img = np.clip(img, 0, 255)
        img = np.uint8(img)

        if verbose:
            cv.imshow("bright_contrast", img)
            #cv.imwrite('../brightContrast'+str(self.count)+'_'+tag+'.png', img)
            k = cv.waitKey(0)
        return img


# Code used to create the ground thruth. The result of this function were manually adjusted to have a good ground thruth for 
# evaluating the quality of the algorithm in the report.
def createGroundThruth():
    path = "../datasets/dataset_2/"
    N_Frame = len([path+name for name in os.listdir(path) if (os.path.isfile(path+name) and ".png" in name) ])

    # First image upload (one frame) in order to find shapes
    ip    = ImageProcessing(path=path)
    for i in range(0, N_Frame):
        img  = ip.acquire_frame()

        img  = ip.crop_image(img, 0, 200, 640, 480)
        #img  = ip.crop_image(img, 0, 120, 640, 480) #ORIGINAL DATASET

        pdf  = ip.create_mask(ip.bright_contr(img,  -60, 100))

        cv.imshow("result", pdf)
        k = cv.waitKey(1)
        #cv.imwrite('../datasets/ground_truth_2/img' + str(i) + '.png', pdf)

#For testing the module
if __name__ == '__main__':
    imgP = ImageProcessing()

    for d in [1,2,3,4]:
        path="../datasets/dataset_"+str(d)+"/"

        files = [name for name in os.listdir(path) if (os.path.isfile(path+name) and ".png" in name)]
        files.sort(key=lambda x: int( (x.split(".")[-2]).split("_")[1] ))
        files = [path + f for f in files]
        for f in files:
            img = cv.imread(f)

            img = imgP.crop_image(img,0, 200, 640, 480)
            img = imgP.bright_contr(img,  brightness = -60,contrast = 127)
            cv.imshow("b_c", img)
            k = cv.waitKey(5)

            img = imgP.create_mask(img, False)
            cv.imshow("mask", img)
            k = cv.waitKey(5)

