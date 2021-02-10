import numpy as np
from scipy import interpolate # Spline usage

debug = False

# Object that represent a single Particle of the Particle filter
class Particle(object):

    def __init__(self, N_points, Interpolation_Points, order, type=None):

        self.N_points = N_points
        self.Interpolation_Points = Interpolation_Points
        self.points=[[0,0]] * self.N_points
        self.order  = order

        self.w_originak = 1  # Particle weight
        self.w          = 1  # Particle weight normalized
        self.spline     = [] # Spline vector

        return

    def toString(self):
        return "Particle: "+ str(self.points) + " Weight: "+ str(self.w)

    def generateSpline(self, offset=0, Interpolation_Points=0):

        spline, x, y   = [], [], []
        if(Interpolation_Points==0):
            Interpolation_Points=self.Interpolation_Points

        if(int(self.order) >= self.N_points):
            print("\norder = " + str(order) + " and N_points = " + str(self.N_points) + " Filter need order < N_points \n")

        for i in range(self.N_points):

            x_temp, y_temp = self.points[i][0] + offset, self.points[i][1]

            x.append(x_temp)
            y.append(y_temp)

        tck, u = interpolate.splprep([x, y], k = int(self.order), s = 0)
        xi_temp, yi_temp = interpolate.splev(np.linspace(0, 1, Interpolation_Points), tck)

        for j in range(len(xi_temp)):
            spline.append([xi_temp[j], yi_temp[j]])

        self.spline = spline
        return self.spline

if __name__ == '__main__':

    p = Particle()
    print(p.toString())

    # point in Particle has two elements, double cicle
    for i in range(0,2):
        p.points[i]=[i+10,i+10]

    p.w = 9
    print(p.toString())
