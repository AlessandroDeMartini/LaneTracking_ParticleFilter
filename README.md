# Lane Detaction and Tracking using a Particle Filter implementation

This repositories contains the project developed by Alessandro De Martini and Simone Morettini.
The code has been developed during the accademic year 2020-2021 under the supervision of Prof. Jhon Folkenson.
The Video of the application is visible at the link: https://youtu.be/z_hctSw-tUc

## Short description

This repository contains a lane tracking algorithm developed using a Particle Filter. The peculiarity of the approach used is that the Particle filter is used both for lane detection and lane tracking. The input of the particle filter is a probability distribution matrix created at the image processing stage. Since the algorithm implemented is simple and uses really few energy it's optimal for being used in small embedded devices. 

Detailed information of the process, the choices made and the results are available on this two papers of the creators: [De Martini's report](https://github.com/AlessandroDeMartini/LaneTracking_ParticleFilter/blob/master/Paper_DeMartini.pdf ) and [Morettini's report](https://github.com/AlessandroDeMartini/LaneTracking_ParticleFilter/blob/master/Paper_Morettini.pdf).
 

## Repositiory content

- src
    - ResultAcquisition: folder with .py files used for results acquisitions
    - ParticleFilter.py: main file for the execution
    - image processing.py: class used for image pre-processing
    - particle.py : object used in particle filter
    - utils.py : class with some drow functions.

- requirements.txt: filed with packages that should be installed for the functionality of the code

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
