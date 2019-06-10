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

COR Predictor CUDA requires Nvidia's CUDA Toolkit, which can be found [here](https://developer.nvidia.com/cuda-downloads). Install the toolkit using the instructions in the [Quick Start Guide](https://docs.nvidia.com/cuda/cuda-quick-start-guide/index.html).

* Gnu+Linux

In the terminal, navigate to the build folder in the parent directory `~/COR-Predictor-CUDA-0.2-beta` with the command 
`~ $ cd COR-Predictor-CUDA-0.2-beta/build`

Run cmake with the command 

`~/COR-Predictor-CUDA-0.2-beta/build $ CUDACXX="NVCC_PATH" cmake -DCMAKE_CUDA_FLAGS="-arch=ARCHITECTURE" ..`

replacing `NVCC_PATH`with the path of the Nvidia Code Compiler, "/usr/local/cuda-10.1/bin/nvcc" for example, and replacing `ARCHITECTURE` with your Nvidia GPU architechture, such as "sm_50".

Make the executable with 

`~/COR-Predictor-0.2-beta/build $ make`

The executable will be created in the build directory. 

Run COR Predictor by entering the command

`~/COR-Predictor-0.2-beta/build $ ./COR-Predictor`

* Windows

In the command line, enter the command

`Run cmake-gui.exe`

or run the program from the start menu, Program Files, or a desktop shortcut.

In the top entry of the GUI, enter the location of the build folder.
No configuration is necessary. Simply press 'Generate' to create the makefile.

---

Using COR Predictor:

COR Predictor uses a genetic algorithm to fit a set of parameters to the training data. The genetic algorithm starts by creating an initial set of parameter configurations and selecting a percentage of them that best fits the training data as a gene pool. These configurations are encoded as 1-d boolean arrays compatable with the GPU.

An iterative process then starts where the best-fitting half of the configurations is selected to be kept for reproduction. Pairs of these configurations are selected as parents, and children configurations are created by carrying over each bit from one of the two parents selected at random. The children replace the configurations not selected for reproduction.

A percentage of all the bits in the entire population is then selected to be mutated by being changed from 1 to 0 or vise versa. This step is important, as it introduces variation to prevent inevitable bottlenecking. Bottlenecking happens because of the finite size of the population, which will eventually reach a point where all the configurations are very similar and stall the process unless variation is continually introduced. In terms of the optimization problem, mutation allows poorer-fitting configurations to be accepted into the population. Thus, mutation is necessary because the objective function space contains numerous local minima that impede searching.

After mutation occurs, the iterative process starts over and repeats until an error tolerance specified by the user is reached. The process of selecting the best-fitting configurations combined with scrambling the bits using reproduction and mutation efficiently searches the objective function space for an optimal configuration.

Settings:

When COR Predictor is run, the program first reads the file `settings.txt` to obtain the settings used to initiate optimization. These settings strongly determine the successfulness of optimization. They must be listed in the file with the syntax

`[Setting_name]=[value]`

If a setting is to be changed, only make changes to the value and leave no space between the value and the `=` sign. The setting name should not be altered, as changing it will have no effect. The setting corresponding to each value is only determined by the order of the listing. 

* `initial_population_size`:

This setting determines the size of the initial population created at initialization. It should be sufficiently large enough to create a wide selection of configurations. If this value is too small, the error of the configurations chosen at startup may be inflated.

* `gene_pool_size`:

This is the number of configurations in the initial population selected to be kept for the iterations. It should be much smaller than the `initial_population_size` setting so that mostly well-fitting configurations will be selected at startup. This setting must also be large enough to slow the speed at which bottlenecking occurs as well as to avoid local minima. `gene_pool_size` is limited by the computer's available processing power, as raising its value scales the number of operations performed on each iteration exponentially. 

* `elite_population_size`:

This setting increases the number of configurations kept in the "elite population", which serves several functions. The elite population is kept safe from mutations that decrease their fitness. This provides protection against divergence, especially at high mutation rates. However, large elite population sizes decrease the speed at which the objective function space is searched. Around 25 percent of the gene pool is an appropriate size.

The elite population also plays a role in reducing the time needed for repeated convergences. At startup, the option is given to use a parameter configuration from a previous convergence, which will be used as each configuration in the elite population. This "jump-starts" the convergence process.

* `mutation_rate`:

The `mutation_rate` setting determines the percentage of bits accross the entire population that are changed during mutation. Increasing its value increases the amount of variation introduced on each iteration and aids in the algorithm's ability to search the objective function space. However, a too-large mutation rate will cause the program to diverge or to scan around wildly without converging. 

* `mean_squared_error`:

This is the mean squares error value at which iterations will conclude. It is better to be conservative (not too small) with this value, as there is no other way to stop the iterative process prematurely at this point. It is better to perform a convergence in multiple steps to avoid the algorithm stalling without reaching the required tolerance. If that happens, the only recourse as of this release is to close out the terminal and restart the program.

Adding Datapoints

COR Predictor reads training datapoints from the files `cor_independent.csv` and `cor_dependent.csv`. To add a datapoint, simply open the files in a spread sheet and enter the data in the next row below the lowest entry in each file. Datapoints can also be entered from the main menu.

Running COR Predictor

COR Predictor is run by entering the command 

`~/COR-Predictor-CUDA-0.2-beta/build $ ./COR-Predictor`

while in the build folder. 

Upon startup, the program reads the files 'cor_independent.csv', 'cor_dependent.csv', 'cor_parameters.csv', and 'settings.txt' automatically. A main menu will appear with four options. The first opens a dialogue that allows the user to enter a new training datapoint. The second option initializes optimization using the settings read from 'settings.txt'. The third opens a dialogue that predicts a coefficient of restitution from material properties. The third option quits the program.

If the second option is chosen, the user will be prompted to choose optimization from a random parameter configuration or from an existing configuration if one exists. The program will then display the mean squared error for each iteration so that the user can track its progress. The main loop will continue until the specified tolerance is reached or the user stops the program. If the required tolerance is reached the program will ask the user whether to write the new parameter configuration to file. If the user responds "y", the new configuration will be saved, overwriting the previous one. If the user responds "n", the new configuration will be discarded. 

Overall, the optimization process could take anywhere from several seconds to several hours depending on the size and quality of the training data. If optimization takes much longer than that, the settings might need to be changed in order for the algorithm to work faster. If optimization stops before reaching the target accuracy, stop the loop manually by pressing 'Enter'.

Troubleshooting

* `settings.txt`/`cor_independent.csv`/`cor_dependent.csv`/`cor_parameters.csv` was not found.

Each of these files must be in the same folder as the executable.

* `cor_independent.csv`/`cor_dependent.csv` is/are the wrong dimension(s).

Each row of `cor_independent.csv` must have 7 columns. Each row of `cor_dependent.csv` must have 1 column.

* `cor_independent.csv` and `cor_dependent.csv` do not have the same number of entries.

Each set of independent variables in `cor_independent.csv` must have a corresponding dependent variable in `cor_dependent.csv` and vice versa. Both files must have the same number of rows.

* The least squares error is very high at startup. 

The fitness of the starting population can be improved somewhat by increasing the size of the initial population.

* The iterations proceed very slowly.

The gene pool size increases the number of operations performed per iteration. For slower computers, the gene pool size should not be larger than 50.

* Population divergence 

If the mutation rate is too high, the mean squared error in each member of the population may fluctuate wildly rather than decreasing overall, or it may increase. The program will stop if it detects that the population is not adapting to the dataset. A reasonable value for the mutation rate is 0.01, though a higher number may be more appropriate for larger datasets with more noise.

* Population bottleneck

Since the number of members in the population is fixed, its diversity decreases as the algorithm progresses until each chromosome is very similar to the others (A Habsburg scenario is guarenteed!). Mutation stops this by introducing new variation to the population, but if the mutation rate is too low the problem will only be delayed. The program will stop if it detects that this state has been reached before reaching the error tolerance. Bottlenecking can be slowed by increasing the gene pool size or increasing the mutation rate.

* The error decreases steadily then decreases progressivly slower.

The mean squared error always decreases very quickly at startup, then slows down as the population begins to converge. When the mean squared error is large, larger mutation rates are favorable to avoid local minima. Once the mean squared error is smaller, the mutation rate should be decreased in order to aid convergence.

---

Credits:

The implementation of the genetic algorithm used in COR Predictor is built off of information available in Tao Pang's "An Introduction to Computational Physics," second edition.

---

Contact:

Contact me at <jball10990@gmail.com>.

