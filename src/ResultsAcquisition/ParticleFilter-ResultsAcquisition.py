import cv2 as cv
import numpy as np
from random import seed
from random import randint
import random
from copy import deepcopy
import math
import os, os.path
import sys
import time
import timeit
import csv

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

'''
Image progessing library import:
- Read from camera
- BW mask
- Blurry
- Crop Image

Particle Object
- Create many particles (Lines)
- Each line is associated with a weight

Utils:
- draw_particles        
- order points          | --> for IPM transformation
- four points transform |
'''

import sys
sys.path.append('../')

from image_processing import ImageProcessing
from particle import Particle
from utils import Utils

test = False
test_weightin_isolated = False
testImageCrop = False

class ParticleFilter(object):

    def __init__(self, N, order, N_points, Interpolation_points, h_image, w_image, type_approximation=None, threshold_reset=None, parameters={}):

        self.particles      = []
        self.cumweights     = []
        self.N              = N
        self.h_image        = h_image
        self.w_image        = w_image

        self.failing_count  = 0
        self.approximation_to_show  = True

        self.order          = order
        self.N_points       = N_points

        self.Interpolation_points = Interpolation_points

        self.approximation  = Particle(self.N_points, self.Interpolation_points, self.order )

        if(type_approximation==None):
            self.type_approximation="max"
        else:
            self.type_approximation=type_approximation

        if(threshold_reset==None):
            self.threshold_reset=5
        else:
            self.threshold_reset=threshold_reset

        return

    def initialization(self, verbose=False):
        # Distribute hundreds of lines over the image
        # Larger the image is covered and more are the chance of find
        # the true line
        h = self.h_image
        w = self.w_image

        # mean and standard deviation for spline generation
        mu, sigma  = w/2, w/3

        for _ in range(0, self.N):

            p = Particle(self.N_points, self.Interpolation_points, self.order)

            # Points distribution, prom the bottom to the top, all the picture is covered
            # Alto mu alto perché gira, con il giro della ruota possiamo aumentare il mu quando gira
            for j in range(self.N_points):

                random_numbers = np.random.normal(mu, sigma, 1).astype(int)
                p.points[j] = [abs(random_numbers[0]), int(j * h/(self.N_points - 1)) ]

            p.w = 1/self.N
            self.particles.append(p)

        return self.particles

    def sampling(self, verbose=False):
        # It is the Prediction -> not use an Encoder, only noise (diffusion step)
        # From the previus step the like marking is maybe changed, we need to follow this change
        # to keep the traking of the line.

        # What we do: we move a random amount to the x coordinate and of start point and
        # and of the end point of the line. the random amount is sampled from a gaussian distribution
        # N(mu, sigma): mu -> current value of x at start or end, sigma

        sigma = [None] * self.N_points

        for i in range(self.N_points):
            sigma[self.N_points - 1 - i] = ((self.N_points-1) - i) * 40/self.N_points + 5

        for p in self.particles:

            if(verbose):
                print("p1_First " + str(p.points))

            for i, point in enumerate(p.points):

                # Variation in X
                rand_norm_1 = np.random.normal(point[0], sigma[i], 1).astype(int)[0]

                # Variation in Y, little
                #rand_norm_2 = np.random.normal(point[1], 1, 1).astype(int)[0]

                point[0] = rand_norm_1
                #point[1] = rand_norm_2

            if(verbose):
                print("p1_After "   + str(p.points))

        return

    def weighting(self, pdf, verbose=False, debug_data=None):

        self.cumweights=[]
        pdf_n=np.array(pdf)

        if(debug_data != None):
            if("particles" in debug_data):
                self.particles=debug_data["particles"]

        tot_w = 0

        for p in self.particles:

            w=0
            p.generateSpline()

            for point_s in p.spline:
                # Each spline point is taken to be weighted
                x, y = int(point_s[0]),int(point_s[1])
                w   += np.sum(pdf_n[y-1:y+2 , x-1:x+2])

            p.w_original = w
            tot_w       += w

        if(verbose):
            print("BEFORE NORMALIZE WEIGHT")
            for p in self.particles:
                print(p.toString())

        tot_tmp=0

        for p in self.particles:
            if(tot_w==0):
                p.w=1/self.N
            else:
                p.w=p.w_original/tot_w
            tot_tmp+=p.w

            # cdf: Cumulative distribution function
            self.cumweights.append(tot_tmp)

        if(verbose):
            print("AFTER NORMALIZE WEIGHT")
            for p in self.particles:
                print(p.toString())

        return

    def resampling(self, verbose=False):

        # It does the resampling using the systematic resampling
        if(verbose):
            print("BEFORE RESAMPLING")
            for p in self.particles:
                print(p.toString())

        new_particles = []
        r_0 = random.uniform(0, 1/self.N)

        for m in range(0, self.N):
            i = next(k for k, value in enumerate(self.cumweights)if value > r_0 + (m-1)/self.N)
            new_particles.append(deepcopy(self.particles[i]))

        self.particles=new_particles
        self.doApproximation()
        # print(str(self.approximation.w_original) +"    <"+ str(0.2*self.Interpolation_points*(9*255)))
        if(self.approximation.w_original < 0.2*self.Interpolation_points*(9*255)): # Line weight < 20% of max weights - 255 max value for the pixel
            self.failing_count+=1
            self.approximation_to_show =False
            if(self.failing_count >= self.threshold_reset):
                    self.failing_count=0
                    self.initialization()
                    #print("reset")
        else:
            self.failing_count=0
            self.approximation_to_show =True

        if(verbose):
            print("AFTER RESAMPLING")
            for p in self.particles:
                print(p.toString())
        return

    def doApproximation(self):
        # Return a particle that approximate the particle distributions
        # It return the average of the particles or the partichle with higher weight
        # type can be "max" or "average"

        if(self.type_approximation=="average"):
            p=deepcopy(self.particles[0])
            for part in self.particles:
                for index, coo in enumerate(p.points):
                    coo[0] += part.points[index][0]
                    coo[1] += part.points[index][1]
            for coo in p.points:
                coo[0] = int(coo[0]/len(self.particles))
                coo[1] = int(coo[1]/len(self.particles))
            if(isinstance(p, Particle)):
                p.generateSpline(Interpolation_points=20)
            self.approximation=p
        elif(self.type_approximation=="max" or True):
            self.approximation = max(self.particles, key=lambda p: p.w)

def execute_test():

    utils = Utils()
    ip    = ImageProcessing()
    pdf, image = ip.get_lines_pdf()

    pf = ParticleFilter(500, image.shape[0],image.shape[1])
    imgP = ImageProcessing()

    p = Particle()
    p.toString()

    # TEST INITIALIZATION
    image_color = cv.cvtColor(pdf, cv.COLOR_GRAY2RGB) # Image with color
    particles = pf.initialization(True)

    utils.draw_particles(image_color, pf.particles, "Initialization")

    ip = ImageProcessing()
    pdf, image= ip.get_lines_pdf()

    verbose=False

    for i in range(0,10):

        # TEST SAMPLING
        pf.sampling(verbose)

        image_color = cv.cvtColor(pdf, cv.COLOR_GRAY2RGB) # Image with color
        utils.draw_particles(image_color, pf.particles, "Sampling")

        # TEST WEIGHTING
        debug_data={}
        pf.weighting(pdf, verbose,  debug_data)

        # TEST RESAMPLING
        pf.resampling(verbose)
        image_color = cv.cvtColor(pdf, cv.COLOR_GRAY2RGB) # Image with color

        utils.draw_particles(image_color, pf.particles, "Reampling")

    cv.destroyAllWindows()

    return

def filter_usage(N_Particles, Interpolation_points, order=2, N_points=3, dataset_number=1, Images_print=False, blur=11, threshold_reset=5):

    resultLeft  = []
    resultRight = []
    utils = Utils()

    pts = []

    path = "../../datasets/dataset_"+str(dataset_number)+"/"
    path_ground_truth = "../../datasets/ground_truth_"+str(dataset_number)+"/"

    save_images = True
    type_approximation   = "max"
    threshold_reset      = threshold_reset
    crop_points        = []
    approximationPF1   = []
    approximationPF2   = []

    if(dataset_number==1):
        crop_points = [0, 120, 640, 480]
        #pts         = np.array([(0, 176-70), (500+100, 186-100), (-148, 345), (639+148, 350)])

    else:
        crop_points = [0, 200, 640, 480]
        #pts         = np.array([(0, 106), (600, 86), (-148, 278), (780, 279)])
        
    # First image upload (one frame) in order to find shapes
    ip    = ImageProcessing(pts, path=path, crop_points=crop_points, kernel_size=blur)
    pdf, pdf1, pdf2, image, image1, image2 = ip.get_lines_pdf()

    if(test):
        execute_test()
        sys.exit()

    if(testImageCrop):
        # Temporarly, when decided the size of image it can be put fixed
        ip    = ImageProcessing(pts)
        pdf, pdf1, pdf2, image, image1, image2 = ip.get_lines_pdf()
        cv.imshow("pdf1", pdf1)
        k = cv.waitKey(0)
        cv.imshow("pdf2", pdf2)
        k = cv.waitKey(0)
        cv.imshow("image1", image1)
        k = cv.waitKey(0)
        cv.imshow("image2", image2)
        k = cv.waitKey(0)

    pf1 = ParticleFilter(N_Particles, order, N_points, Interpolation_points, image1.shape[0], image1.shape[1], type_approximation=type_approximation, threshold_reset=threshold_reset)
    pf2 = ParticleFilter(N_Particles, order, N_points, Interpolation_points, image2.shape[0], image2.shape[1], type_approximation=type_approximation, threshold_reset=threshold_reset)

    ip = ImageProcessing(pts, path, crop_points=crop_points, kernel_size=blur)

    N_STEP = len([path+name for name in os.listdir(path) if (os.path.isfile(path+name) and ".png" in name) ])-2

    pf1.initialization()
    pf2.initialization()

    left_line_found=0
    right_line_found=0

    for step in range(0, N_STEP):

        pf1.sampling()
        pf2.sampling()

        pdf, pdf1, pdf2, image, image1, image2 = ip.get_lines_pdf()

        pf1.weighting(pdf1)
        pf2.weighting(pdf2)

        pf1.resampling()
        pf2.resampling()

        approximationPF1.append(pf1.approximation)
        approximationPF2.append(pf2.approximation)

        if(Images_print):
            image_color  = cv.cvtColor(pdf, cv.COLOR_GRAY2RGB)  # Image with color
            image_color1 = cv.cvtColor(pdf1, cv.COLOR_GRAY2RGB) # Image with color
            image_color2 = cv.cvtColor(pdf2, cv.COLOR_GRAY2RGB) # Image with color

        best_particles       = []
        offset_Approximation = []

        if(pf1.approximation_to_show):
            best_particles.append(pf1.approximation)
            offset_Approximation.append(0)

        if(pf2.approximation_to_show):
            best_particles.append(pf2.approximation)
            offset_Approximation.append(int(image.shape[1]/2 ))

        if(Images_print):
            #utils.draw_particles(image_color2, pf2.particles, [pf2.approximation], "Reampling_PDF2")
            #pdf_truth1_color= cv.cvtColor(pdf_truth1, cv.COLOR_GRAY2RGB)
            #utils.draw_particles(pdf_truth1_color, pf1.particles, [pf1.approximation], "GROUND TRUTH")

            res_1 = utils.draw_particles(image, [], best_particles, "Resampling", offset=[0]*len(pf1.particles)+[int(image.shape[1]/2 )]*len(pf2.particles), offsetApproximation=offset_Approximation)
            res_2 = utils.draw_particles(image_color, pf1.particles + pf2.particles, best_particles, "Resampling_PDF", offset=[0]*len(pf1.particles)+[int(image.shape[1]/2 )]*len(pf2.particles), offsetApproximation=offset_Approximation)

            if(save_images ):
                cv.imwrite('../../outputs/particles_output/img_' + str(step) + '.png', res_2)
                cv.imwrite('../../outputs/result_output/img' + str(step) + '.png', res_1)

    return approximationPF1, approximationPF2, N_STEP

def accuracy_computation(approximationPF1, approximationPF2, dataset_number=1, raw=False):

    utils = Utils()
    resultLeft       = []
    resultRight      = []
    resultAvg       = []
    crop_points      = []
    pts              = []
    left_line_found  = 0
    right_line_found = 0

    path_ground_truth = "../../datasets/ground_truth_"+str(dataset_number)+"/"
    N_STEP = len(approximationPF1)

    if(dataset_number==1):
        crop_points = [0, 120, 640, 480]
        #pts         = np.array([(0, 176-70), (500+100, 186-100), (-148, 345), (639+148, 350)])
    else:
        crop_points = [0, 200, 640, 480]
        #pts         = np.array([(0, 106), (600, 86), (-148, 278), (780, 279)])

    ip_truth               = ImageProcessing(pts, path=path_ground_truth, crop_points=crop_points)

    for step in range(0, N_STEP):

        pdf_truth1, pdf_truth2 = ip_truth.get_raw_image()

        # pdf_truth1_color= cv.cvtColor(pdf_truth1, cv.COLOR_GRAY2RGB)
        # utils.draw_particles(pdf_truth1_color, [], [approximationPF1[step]], "GROUND TRUTH")

        acc1 = utils.evaluate(approximationPF1[step], pdf_truth1)
        acc2 = utils.evaluate(approximationPF2[step], pdf_truth2)

        resultLeft.append(acc1)
        resultRight.append(acc2)
        resultAvg.append((acc1+acc2)/2)
        left_line_found += 1 if acc1 > 0.5 else 0
        right_line_found += 1 if acc2 > 0.5 else 0

    Avarage_accuracy   = (np.average(resultLeft) + np.average(resultRight))/2
    Standard_deviation = (np.std(resultLeft) + np.std(resultRight))/2
    Line_Found         = (left_line_found/N_STEP + right_line_found/N_STEP)/2

    print("Average accuracy  : LEFT: " + str(np.average(resultLeft)) + " RIGHT: " + str(np.average(resultRight)) + " AVARAGE: " + str(Avarage_accuracy) )
    print("Standard deviation: LEFT: " + str(np.std(resultLeft)) + " RIGHT: " + str(np.std(resultRight)) + " AVARAGE: " + str(Standard_deviation))
    print("Percentage of line: LEFT: " + str(left_line_found/N_STEP) + " RIGHT: " + str(right_line_found/N_STEP) + " AVARAGE: " + str(Line_Found))

    if(raw):
        # resultAvg find the avarage for each Image frame
        return Avarage_accuracy, Standard_deviation, Line_Found, resultAvg
    else:
        return Avarage_accuracy, Standard_deviation, Line_Found

def randrange(n, vmin, vmax):
    '''
    Helper function to make an array of random numbers having shape (n, )
    with each number distributed Uniform(vmin, vmax).
    '''
    return (vmax - vmin)*np.random.rand(n) + vmin

if __name__ == '__main__':

    Timing              = False
    graph               = False
    blur_tuning         = True
    threshold           = False
    accuracy_over_time  = False

    if(Timing):

        Particles_Number = []
        Time_spent       = []
        Accuracy         = []
        St_dev           = []
        Ln_Fnd           = []

        for i in range(100, 200, 10):

            start_time  = time.process_time()
            approximationPF1, approximationPF2, N_frame = filter_usage(50, i, False)
            tot_time = time.process_time() - start_time

            Avarage_accuracy, Standard_deviation, Line_Found = accuracy_computation(approximationPF1, approximationPF2)

            print(tot_time)

            Time_spent.append(tot_time/N_frame)
            Particles_Number.append(i)
            Accuracy.append(Avarage_accuracy)
            St_dev.append(Standard_deviation)
            Ln_Fnd.append(Line_Found)

            with open("../time_dataset1_NInt-50PARTc.csv", "w") as file:
                writer = csv.writer(file)
                writer.writerow(Time_spent)
                writer.writerow(Particles_Number)
                writer.writerow(Accuracy)
                writer.writerow(St_dev)
                writer.writerow(Ln_Fnd)

    elif(graph):

        plot_3D = False

        with open("../../Results/time_dataset2_NSAMPLE-20Int.csv", newline='') as file:
            data_Particles = list(csv.reader(file))

        Particles_timing      = [float(s) for s in data_Particles[0] ]
        Particles_N_particles = [float(s) for s in data_Particles[1] ]
        Particles_Accuracy    = [float(s) for s in data_Particles[2] ]
        Particles_St_dev      = [float(s) for s in data_Particles[3] ]
        Particles_Ln_Fnd      = [float(s) for s in data_Particles[4] ]

        with open("../../Results/time_dataset2_NInt-50Samples.csv", newline='') as file:
            data_Interp = list(csv.reader(file))

        Interp_timing   = [float(s) for s in data_Interp[0] ]
        Interp_N_Interp = [float(s) for s in data_Interp[1] ]
        Interp_Accuracy = [float(s) for s in data_Interp[2] ]
        Interp_St_dev   = [float(s) for s in data_Interp[3] ]
        Interp_Ln_Fnd   = [float(s) for s in data_Interp[4] ]

        if(plot_3D):

            fig_Particles, fig_Interp = plt.figure(),  plt.figure()
            ax_Particles, ax_Interp   = fig_Particles.add_subplot(111, projection='3d'), fig_Interp.add_subplot(111, projection='3d')

            x_Particles, y_Particles, z_Particles = Particles_N_particles, Particles_Accuracy, Particles_timing
            x_Interp   , y_Interp   , z_Interp    = Interp_N_Interp,       Interp_Accuracy,    Interp_timing

            n_Particles, n_Interp = len(Particles_timing), len(Interp_timing)

            for c, m, zlow, zhigh in [('r', 'o', -50, -25), ('b', '^', -30, -5)]:
                ax_Particles.scatter(x_Particles, y_Particles, z_Particles, c=c, marker=m)
                ax_Interp.scatter(x_Interp   , y_Interp   , z_Interp   , c=c, marker=m)

            ax_Particles.set_xlabel(' N_particles ')
            ax_Particles.set_ylabel(' Accuracy ')
            ax_Particles.set_zlabel(' Time ')

            ax_Interp.set_xlabel(' N_Int_points ')
            ax_Interp.set_ylabel(' Accuracy ')
            ax_Interp.set_zlabel(' Time ')

            plt.show()

        else:
            x_Particles, y_Particles, z_Particles = Particles_N_particles[0:40], Particles_Accuracy[0:40], Particles_timing[0:40]
            x_Interp   , y_Interp   , z_Interp    = Interp_N_Interp[0:40], Interp_Accuracy[0:40], Interp_timing[0:40]

            plt.plot(x_Particles, z_Particles, 'b-')
            plt.plot(x_Interp   , z_Interp   , 'r-')

            plt.ylabel('Time/frame')

            plt.legend( [ 'Number of particles'         # This is f(x)
                        , 'Number of spline interpolation points' # This is g(x)
                        ] )
            plt.title('Number selection for particles and spline interpolation points - Dataset 1');

            plt.show()

    elif(blur_tuning):
        #Try different types of Blue
        Blur_kernel_size = [3, 5, 7, 9, 11, 15, 17, 19]
        Accuracy         = []
        St_dev           = []
        Ln_Fnd           = []

        avg_accuracy_datasets=[]
        avg_std_dev_datasets=[]
        avg_lines_datasets=[]
        for k_s in Blur_kernel_size:
            for dataset_number in [1,2,3]:
                approximationPF1, approximationPF2, N_frame = filter_usage(56, 11, Images_print=False, blur=k_s, dataset_number=dataset_number)
                Avarage_accuracy, Standard_deviation, Line_Found = accuracy_computation(approximationPF1, approximationPF2, dataset_number=dataset_number)
                avg_accuracy_datasets.append(Avarage_accuracy)
                avg_std_dev_datasets.append(Standard_deviation)
                avg_lines_datasets.append(Line_Found)
            if(Blur_kernel_size == 19 or Blur_kernel_size == 3):
                time.sleep(10)
            Accuracy.append(np.average(Avarage_accuracy))
            St_dev.append(np.average(Standard_deviation))
            Ln_Fnd.append(np.average(Line_Found))
            print("Kernel size "+str(k_s)+" Accuracy: "+str(Accuracy[-1])+" Std.dev.: "+str(St_dev[-1])+" Line Found: "+str(Ln_Fnd[-1]))

        # with open("../test_NOTUSE_blurnoisy_dataset1_11Int-56PART.csv", "w") as file:
        #     writer = csv.writer(file)
        #     writer.writerow(Blur_kernel_size)
        #     writer.writerow(Accuracy)
        #     writer.writerow(St_dev)
        #     writer.writerow(Ln_Fnd)

    elif(threshold):
        Threshold_Value  = []
        Time_spent       = []
        Accuracy         = []
        St_dev           = []
        Ln_Fnd           = []
        dataset_number   = 2

        for i in range(1, 11):
            approximationPF1, approximationPF2, N_frame = filter_usage(50, 17, Images_print=False, threshold_reset=i,dataset_number=dataset_number)
            Avarage_accuracy, Standard_deviation, Line_Found = accuracy_computation(approximationPF1, approximationPF2, dataset_number=dataset_number)

            Threshold_Value.append(i)
            Accuracy.append(Avarage_accuracy)
            St_dev.append(Standard_deviation)
            Ln_Fnd.append(Line_Found)

            with open("../threshold_dataset2_17Int-50PART.csv", "w") as file:
                writer = csv.writer(file)
                writer.writerow(Threshold_Value)
                writer.writerow(Accuracy)
                writer.writerow(St_dev)
                writer.writerow(Ln_Fnd)
    
    elif(accuracy_over_time):
        #Different types of complexity
        
        complexity  = [[50,17],[100,25]]
        Accuracy    = []
        St_dev      = []
        Ln_Fnd      = []

        #avg_accuracy_datasets=[]
        #avg_std_dev_datasets=[]
        #avg_lines_datasets=[]

        accuracies=[]
        for dataset_number in [1,2,3,4]:
            print("dataset" + str(dataset_number))
            for c in complexity:

                approximationPF1, approximationPF2, N_frame = filter_usage(c[0], c[1], Images_print=False, dataset_number=dataset_number)
                Avarage_accuracy, Standard_deviation, Line_Found, Accuracy_list = accuracy_computation(approximationPF1, approximationPF2, dataset_number=dataset_number, raw=True)
                print(Avarage_accuracy)
                print(Accuracy_list)
                #avg_accuracy_datasets.append(Avarage_accuracy)
                #avg_std_dev_datasets.append(Standard_deviation)
                #avg_lines_datasets.append(Line_Found)
                #Accuracy.append(np.average(Avarage_accuracy))
                #St_dev.append(np.average(Standard_deviation))
                #Ln_Fnd.append(np.average(Line_Found))
                save = False
                if(save):
                    with open("../accuracies_dataset_"+str(dataset_number)+"NPart"+str(c[0])+"NInt"+str(c[1])+".csv", "w") as file:
                        writer = csv.writer(file)
                        #writer.writerow(index)
                        #writer.writerow(Accuracy)
                        #writer.writerow(St_dev)
                        writer.writerow(Accuracy_list)
    else:

        N_particles           = 56
        Interpolation_points  = 11
        order                 = 3
        N_points              = 4
        dataset_number        = 1

        approximationPF1, approximationPF2, N_frame      = filter_usage(N_particles, Interpolation_points, order, N_points, dataset_number, Images_print=True)
        Avarage_accuracy, Standard_deviation, Line_Found, = accuracy_computation(approximationPF1, approximationPF2, dataset_number)

        '''
        print("Dataset 1")
        dataset_number       = 1
        N_particles          = 56
        Interpolation_points = 11

        print("Linear, 2 points")
        approximationPF1, approximationPF2, N_frame      = filter_usage(N_particles, Interpolation_points, order=1, N_points=2, dataset_number=dataset_number, Images_print=False)
        Avarage_accuracy, Standard_deviation, Line_Found = accuracy_computation(approximationPF1, approximationPF2, dataset_number=dataset_number)
        print("Linear, 3 points")
        approximationPF1, approximationPF2, N_frame      = filter_usage(N_particles, Interpolation_points, order=1, N_points=3, dataset_number=dataset_number, Images_print=False)
        Avarage_accuracy, Standard_deviation, Line_Found = accuracy_computation(approximationPF1, approximationPF2, dataset_number=dataset_number)
        print("Linear, 4 points")
        approximationPF1, approximationPF2, N_frame      = filter_usage(N_particles, Interpolation_points, order=1, N_points=4, dataset_number=dataset_number, Images_print=False)
        Avarage_accuracy, Standard_deviation, Line_Found = accuracy_computation(approximationPF1, approximationPF2, dataset_number=dataset_number)
        print("Quadratic, 3 points")
        approximationPF1, approximationPF2, N_frame      = filter_usage(N_particles, Interpolation_points, order=2, N_points=3, dataset_number=dataset_number, Images_print=False)
        Avarage_accuracy, Standard_deviation, Line_Found = accuracy_computation(approximationPF1, approximationPF2, dataset_number=dataset_number)
        print("Quadratic, 4 points")
        approximationPF1, approximationPF2, N_frame      = filter_usage(N_particles, Interpolation_points, order=2, N_points=4, dataset_number=dataset_number,Images_print=False)
        Avarage_accuracy, Standard_deviation, Line_Found = accuracy_computation(approximationPF1, approximationPF2, dataset_number=dataset_number)
        print("Cubic, 4 points")
        approximationPF1, approximationPF2, N_frame      = filter_usage(N_particles, Interpolation_points, order=3, N_points=4, dataset_number=dataset_number, Images_print=False)
        Avarage_accuracy, Standard_deviation, Line_Found = accuracy_computation(approximationPF1, approximationPF2, dataset_number=dataset_number)

        
        print("Dataset 2")
        dataset_number       = 1
        N_particles          = 50
        Interpolation_points = 17
        
        print("Linear, 2 points")
        approximationPF1, approximationPF2, N_frame      = filter_usage(N_particles, Interpolation_points, order=1, N_points=2, dataset_number=dataset_number, Images_print=False)
        Avarage_accuracy, Standard_deviation, Line_Found = accuracy_computation(approximationPF1, approximationPF2, dataset_number=dataset_number)
        print("Linear, 3 points")
        approximationPF1, approximationPF2, N_frame      = filter_usage(N_particles, Interpolation_points, order=1, N_points=3, dataset_number=dataset_number, Images_print=False)
        Avarage_accuracy, Standard_deviation, Line_Found = accuracy_computation(approximationPF1, approximationPF2, dataset_number=dataset_number)
        print("Linear, 4 points")
        approximationPF1, approximationPF2, N_frame      = filter_usage(N_particles, Interpolation_points, order=1, N_points=4, dataset_number=dataset_number, Images_print=False)
        Avarage_accuracy, Standard_deviation, Line_Found = accuracy_computation(approximationPF1, approximationPF2, dataset_number=dataset_number)
        print("Quadratic, 3 points")
        approximationPF1, approximationPF2, N_frame      = filter_usage(N_particles, Interpolation_points, order=2, N_points=3, dataset_number=dataset_number, Images_print=False)
        Avarage_accuracy, Standard_deviation, Line_Found = accuracy_computation(approximationPF1, approximationPF2, dataset_number=dataset_number)
        print("Quadratic, 4 points")
        approximationPF1, approximationPF2, N_frame      = filter_usage(N_particles, Interpolation_points, order=2, N_points=4, dataset_number=dataset_number,Images_print=False)
        Avarage_accuracy, Standard_deviation, Line_Found = accuracy_computation(approximationPF1, approximationPF2, dataset_number=dataset_number)
        print("Cubic, 4 points")
        approximationPF1, approximationPF2, N_frame      = filter_usage(N_particles, Interpolation_points, order=3, N_points=4, dataset_number=dataset_number, Images_print=False)
        Avarage_accuracy, Standard_deviation, Line_Found = accuracy_computation(approximationPF1, approximationPF2, dataset_number=dataset_number)
        '''
        



