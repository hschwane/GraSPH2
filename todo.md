# my todo list

## high priority

### particle management

- check for valid parameters on all template calls
- save make functions for all buffers (like the one the particle has)
- more safety / better errors when dealing with particle buffers

- think about the particle management carefully
- think about particles during init carefully

- use the new particle things in a correct way

### simulation management
- find better solution for the stuff in the algorithms file


## medium priority

### particle creation
- add 2d image source from the other branch
- make modifiers work
- add modifieres
- add sources
- add constructors for uniform sphere

### usability
- more runtime options vs compiletime options

### simulation management
- make the compute derivitives function more organized
- find better solution for 2D simulation

### integration
- automatically integrate what is needed using TMP

### performance
- somehow automatically adjust particle attributes based on options
- optimize the functions in models.h

### simulation features
- add functionality from the other branch

### output data handling
- use pinned host memory here
- remove the stream sync points
- cuda memcopy is not async


## low priority / ideas / long term projects

### particle creation
- add python script support to generate initials

### particle management
- enhance compile time when using particle templates

### simulation management
- find better solution to switch options on and off

### integration
- better structure to different integrators
- adaptive algorithms

### performance
- add datastructures
- try new graph style kernel launching
- MPI support
- share load between CPU and GPU

### simulation features
- linear consistency fix
- investigate angular momentum (not)conservation
- add collisional merging simulation
- different materials (using recursive kernel launches)

### output data handling
- provide binary data format
- provide python script to generate 3d html plots



# finished
for motivation, all finished todo entries are moved here instead of being deleted

- fix initialisation function of particle buffer
- mapped and pinned state of resources is not overwritten after most assignments
- device particles can now be accessed from host code
- pinned host memory can be used for host particle buffer
- add function to concatenate and merge particles
- derivatives are now marked as such in the particle buffer

- better defaults for particle load
- add shared buffer
- implement initialisation of device particles again
- somehow make it possible to get the particle type from a base buffer
- somehow make it possible to get the different buffer types from each other
- add reference buffer
- add device reference
- add device buffer

- make it easier to use Particles in the correct way
- safety measures when creating particles and buffers
- better error messages when using Particles and buffers

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
