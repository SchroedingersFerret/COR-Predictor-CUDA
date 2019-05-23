COR Predictor CUDA
=============
version 0.2-beta

By: J. Ball (SchroedingersFerret)

README

COR Predictor CUDA is an improvement on my existing COR Predictor program that takes advantage of Nvidia's CUDA parallel computing system. COR Predictor's optimization scheme relies on the genetic algorithm, which is considered to be "embarassingly parallel". This is because the algorithm acts on each member of the gene pool independently of all other members, and there is no issue in working the algorithm on each member simultaneously. 

---

Files:

In addition to the executable, COR Predictor makes use of the following files:
 
`cor_independent.csv`:

The file `cor_independent.csv` is an n x 7 table containing the independent variables for n training datapoints. The variables represent two sets of three different physical quantities, one for each of two colliding objects, plus the relative velocity of the objects. 


* Yield Strength:

The first quantity, stored in `independent[i][0]` and `independent[i][1]`, is the yield strength in GPa of objects 1 and 2 respectively. _Yield strength_ here is defined as the limit of elastic behavior, beyond which the stress causes the object to deform plastically. 

* Young's Modulus

The second quantities are stored in `independent[i][2]` and `independent[i][3]` and are the Young's modulus in GPa of objects 1 and 2 respectively. 

* Density:

`independent[i][4]` and `independent[i][5]` are the densities in Mg/m^3 of objects 1 and 2 respectively. 

* Velocity: 

`independent[i][6]` is the magnitude of the objects' relative velocity.


`cor_dependent.csv`:

The file `cor_dependent.csv` is an n x 1 table containing the dependent variables for n training datapoints. Each variable is the COR for the collision represented by each datapoint.


`settings.txt`:

File `settings.txt` contains the settings used in the genetic searching algorithm used by the learning program. The various entries and their functions are covered in the section titled "Using COR Predictor".

`cor_parameters.csv`:

This file is generated at the end of optimization and contains the best-fitting parameter configuration. It can be used at startup to reduce the time needed for further optimization.

---

Building COR Predictor CUDA:

COR Predictor CUDA requires Nvidia's CUDA Toolkit, which can be found [here] 
[toolkit] https://developer.nvidia.com/cuda-downloads
