# EL2320 Applied Estimation, Project

This repositories contains the project developed by Alessandro De Martini and Simone Morettini.

The code has been developed during the accademic year 2020-2021 under the supervision of Prof. Jhon Folkenson.

The Video of the application is visible at the link: https://youtu.be/z_hctSw-tUc

## Repositiory content__

- datasets: contains all the datasets needed for the particle filter execution. The dataset are four and they are collected using a car model 1:10 scale. Each datasets contains something different and they are all usefull for testing all the paramether of the filter

- results: folder with results
    - Images
    - .csv/.txt files

- src
    - ResultAcquisition: folder with .py files used for results acquisitions
    - ParticleFilter.py: main file for the execution
    - image processing.py: class used for image pre-processing
    - particle.py : object used in particle filter
    - utils.py : class with some drow functions.

- venv: virtual environment folder

- requirements.txt: filed with packages that should be installed for the functionality of the code

### Dataset description

- dataset_1: straight continous line
- dataset_2: straight segmented line + change of lane + straightfroward and backforward
- dataset_3: bend + cross road
- dataset_4: bend + cross road with light reflection


### Result Folder

The _Result_ contains some of the most important result derived from the particle filter usage. Going throw them is possible to understand some of the choises made for the filter. Moreover the general performance of the filter could be analized goind throw the document inside.

__.csv files__

Many csv files has been acqured, the are the source for many of the images

- time_dataset... (4 files) : these four files contain 5 rows for each number of particles or interpolation points
    1. Time_s pent                        → time spent for each frame
    2. Particles_Number/Int_point_number  → number of particles or interpolation points
    3. Accuracy                           → accuracy for this number
    4. St_dev                             → standard deviation for this number
    5. Ln_Fnd                             → persentage of line found

- threshold... (2 files) : these two files contains 4 rows for each treshold value chosen (when do we reset the filter - Initialization again). 
    1. Treshold  → value of the treshold
    2. Accuracy  → number of particles or interpolation points
    3. St_dev    → standard deviation for this number
    4. Ln_Fnd    → persentage of line found

- blur... (3 files) : Blur is represented as a function Blur(altezza, larghezza). Some results based on different level of blur has been collected. 
    1. blur_      → Blur(value*3, value) with better configuration of brightness and contrast: brightness: -60, contrast: 100
    2. blur7_     → Blur(value*7, value) with better configuration for brightness and contrast. brightness:-60, contrast: 100 _Not really better than before, almost the same_
    3. blurnoisy_ → Blur(value*3, value) with a value of brightness and contrast which create a more noise image: brightness: -120, contrast: 127. Now the size make difference. That happen because bright and contrast remove a lot of noise and also good points so few white points remain. So a bigger blur help to spread the white. Unfortunatelly the differense wrt bebore is small 

- accuracies... (8 files): files created analizing the filter performace on each frame for each dataset. the percormance have been analized considered a slow filter (more particles, more interpolation points) and a fast filter (less particles, less interpolation points). Only one line is present and it is refered to the accuracy on each frame


__.pdf files__

- Figure "ParSelection_Dataset_1" and "ParSelection_Dataset_2" and are refered to time_dataset.. files → 4 files
- Figure "Treshold1" and "Treshold2"  are refered to threshold... files → 2 files.
- Figure "AccTime_Dataset_1", "AccTime_Dataset_2", "AccTime_Dataset_3", "AccTime_Dataset_4" are referred to accuracies... files → (8 files). Each image contains both the fast and the slaw filter graph.

__.txt files__

The text file collect the filter performance considering different particles' shape. Each particle shape have been analized also considering a bird eye view. Not significant results have been analyzed. Using the .txt data two tables could be filled with the accuracy.

|   No IPM  | 2 points | 3 points | 4 points |   |   |    IPM    | 2 points | 3 points | 4 points |
|:---------:|:--------:|:--------:|:--------:|---|---|:---------:|----------|----------|----------|
| Linear    |   *****  |   *****  |   *****  |   |   | Linear    |   *****  |   *****  |   *****  |
| Quadratic |          |   *****  |   *****  |   |   | Quadratic |          |   *****  |   *****  |
| Cubic     |          |          |   *****  |   |   | Cubic     |          |          |   *****  |


### Code Usage 

__Linux - MAC__

- Virtualenv Usage: permits to install not permanentelly all the packages necessaries for the code execution

_First installation_<br/>
(proj repos.) `virtualenv -p python3 venv`

_Activate Virtualenv_ → always before execution<br/>
(proj repos.) `source venv/bin/activate`

_Install requirements pakages_<br/>
(proj repos.) `pip install -r requirements.txt`

- Code Execution from<br/>
(src folder) `python3 ParticleFilter.py` main file execution, all the other file will be read from it.

__Windows__

Create environment `py -m venv venv`

Activate environment `.\venv\Scripts\activate`

Install dependencies `pip install -r ./requirements.txt`

- Code Execution from<br/>
(src folder) `python3 ParticleFilter.py` main file execution, all the other file will be read from it.