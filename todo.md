# my todo list

## high priority

### bugfixes
- fix initialisation of host particle buffer

### particle management
- somehow make it possible to get the host type from a base buffer
- more safety / better errors when dealing with particles and bufffer
- think about the particle management carefully
- think about particles during init carefully
- enhance compile time using particle

### simulation management
- find better solution for the stuff in the algorithms file

## medium priority

### particle creation
- make modifiers work
- add modifieres
- add sources
- add constructors for uniform sphere
- add 2d image source from the other branch
- add python script support to generate initials

### usability
- more runtime options vs compiletime options

### simulation managemant
- make the compute derivitives function more organized
- find better solution to switch options on and off
- find better solution for 2D simulation

### simulation features
- add functionality from the other branch
- linear consistency fix
- investigate angular momentum (not)conservation
- add collisional merging simulation
- different materials (using recursive kernel launches)

### integration
- better structure to different integrators
- adaptive algorithms

### performance
- somehow automatically adjust particle attributes based on options
- optimize the functions in models.h
- add datastructures
- try new graph style kernel launching
- MPI support
- share load between CPU and GPU

### output data handling
- use pinned host memory here
- remove the stream sync points
- provide binary data format
- provide python script to generate 3d html plots


# finished
for motivation, all finished todo entries are moved here instead of being deleted

- add function to concatenate particles
- mark deriviatives in the particle buffer
- better error messages when instantiating a particle wrong

--- v0.2.2 --- 17.12.2018
- use different cuda stream for downloading data
- fix opengl frontend camera in mpUtils
- use logFlush for cuda assert in mpUtils
- add output operator for cuda data types in mpUtils
- fix cmake bug with multiple cuda gencode arguments in mpUtils
- keys for some rendering settings in openGL mode
- runtime coloring switches for openGL
- resize callback for openGL
- fix particle numbers that are not power of 2
- write installation instructions

--- v0.2.1 --- 13.12.2018
- add example SLURM job script
- fix compilation with older cmake 1.10

--- v0.2.0 --- 12.12.2018
- add some kind of structure to the deriviative computation
- make all settings from the settings file actually work
- add functionality to the console frontend
- add some more feedback to the frontends
- add limit of pending copy operations for output data handling
- add possibility to dump simulation results to file
- add source to generate from file
- make the demo simulation from the settings file work with the uniform sphere generator
- add some kind of setting file
- start cleaning the main.cu file
- rewrite build system
- reorder project files
