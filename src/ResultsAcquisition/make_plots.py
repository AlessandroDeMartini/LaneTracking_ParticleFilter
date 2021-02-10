##Make plots
import numpy as np
import matplotlib.pyplot as plt
import csv

## HISTOGRAMS

# Dataset 2 ----------------------------------------------------------------------------------------------------
plt.figure()
height_1 = [0.28443478260869565,0.26960869565217394,0.26821739130434785,0.28030434782608693,0.2714347826086956,0.2619565217391304,0.25469565217391305,0.2554782608695652,0.27708695652173915,0.26160869565217393]
bars_1   = [1,2,3,4,5,6,7,8,9,10]
y_pos_1  = np.arange(len(bars_1))
plt.bar(y_pos_1, height_1, color = 'blue')

# Custom Axis title
plt.xlabel('Reinitialization steps')#, fontweight='bold', color = 'orange', fontsize='17', horizontalalignment='center')
plt.ylabel('Accuracy')
plt.title(' Accuracy given different treschold for filter reset - Dataset 1')


# ---------------------------------------------------------------------------------------------------------------
# Dataset 2 -----------------------------------------------------------------------------------------------------
plt.figure()
height_2 = [0.29125,0.2756122448979592,0.2629974489795919,0.2584311224489796,0.2385841836734694,0.2563520408163265,0.24822704081632652,0.25631377551020407,0.2490433673469388,0.24710459183673472]
bars_2   = [1,2,3,4,5,6,7,8,9,10]
y_pos_2  = np.arange(len(bars_2))
plt.bar(y_pos_2, height_2, color = 'blue')

# Custom Axis title
plt.xlabel('Reinitialization steps')#, fontweight='bold', color = 'orange', fontsize='17', horizontalalignment='center')
plt.ylabel('Accuracy')
plt.title(' Accuracy given different treschold for filter reset - Dataset 2')

plt.show()
# ---------------------------------------------------------------------------------------------------------------

with open("../accuracies_dataset_1NPart50NInt17.csv", newline='') as file:
    acc_1_50_17 = list(csv.reader(file))
with open("../accuracies_dataset_1NPart100NInt25.csv", newline='') as file:
    acc_1_100_25 = list(csv.reader(file))

with open("../accuracies_dataset_2NPart50NInt17.csv", newline='') as file:
    acc_2_50_17 = list(csv.reader(file))
with open("../accuracies_dataset_2NPart100NInt25.csv", newline='') as file:
    acc_2_100_25 = list(csv.reader(file))

with open("../accuracies_dataset_3NPart50NInt17.csv", newline='') as file:
    acc_3_50_17 = list(csv.reader(file))
with open("../accuracies_dataset_3NPart100NInt25.csv", newline='') as file:
    acc_3_100_25 = list(csv.reader(file))

with open("../accuracies_dataset_4NPart50NInt17.csv", newline='') as file:
    acc_4_50_17 = list(csv.reader(file))
with open("../accuracies_dataset_4NPart100NInt25.csv", newline='') as file:
    acc_4_100_25 = list(csv.reader(file))

acc_1_50_17   = [float(s) for s in acc_1_50_17[0] ]
acc_1_100_25  = [float(s) for s in acc_1_100_25[0] ]
acc_2_50_17   = [float(s) for s in acc_2_50_17[0] ]
acc_2_100_25  = [float(s) for s in acc_2_100_25[0] ]
acc_3_50_17   = [float(s) for s in acc_3_50_17[0] ]
acc_3_100_25  = [float(s) for s in acc_3_100_25[0] ]
acc_4_50_17   = [float(s) for s in acc_4_50_17[0] ]
acc_4_100_25  = [float(s) for s in acc_4_100_25[0] ]


######DATASET 1
plt.figure()
plt.plot([i for i in range(0, len(acc_1_50_17))], acc_1_50_17, 'b-')
plt.plot([i for i in range(0, len(acc_1_100_25))], acc_1_100_25, 'r-')

plt.ylabel('Accuracy')
plt.xlabel('Frame')

plt.legend( [ 'Fast algorithm'         # This is f(x)
            , 'Slower algorithm' # This is g(x)
            ] )
plt.title('Accuracy through time - Dataset 1')

######DATASET 2
plt.figure()
plt.plot([i for i in range(0, len(acc_2_50_17))], acc_2_50_17, 'b-')
plt.plot([i for i in range(0, len(acc_2_100_25))], acc_2_100_25   , 'r-')

plt.ylabel('Accuracy')
plt.xlabel('Frame')

plt.legend( [ 'Fast algorithm' # This is f(x)
            , 'Slow algorithm' # This is g(x)
            ] )
plt.title('Accuracy through time - Dataset 2')

######DATASET 3
plt.figure()
plt.plot([i for i in range(0, len(acc_3_50_17))], acc_3_50_17, 'b-')
plt.plot([i for i in range(0, len(acc_3_100_25))], acc_3_100_25   , 'r-')

plt.ylabel('Accuracy')
plt.xlabel('Frame')

plt.legend( [ 'Fast algorithm'   # This is f(x)
            , 'Slow algorithm' # This is g(x)
            ] )
plt.title('Accuracy through time - Dataset 3')

######DATASET 4
plt.figure()
plt.plot([i for i in range(0, len(acc_4_50_17))], acc_4_50_17, 'b-')
plt.plot([i for i in range(0, len(acc_4_100_25))], acc_4_100_25   , 'r-')

plt.ylabel('Accuracy')
plt.xlabel('Frame')

plt.legend( [ 'Fast algorithm'         # This is f(x)
            , 'Slower algorithm' # This is g(x)
            ] )
plt.title('Accuracy through time - Dataset 4')

plt.show()