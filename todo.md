# my todo list

## high priority

## medium priority

### bugfixes
- fix initialisation of host particle buffer

### particle management
- somehow make it possible to get the host type from a base buffer
- think about the particle management carefully
- think about particles during init carefully
- enhance compile time using particle

### particle creation
- make modifiers work
- add modifieres
- add sources
- add constructors for uniform sphere
- add 2d image source from the other branch

### frontend
- runtime coloring switches on openGL
- allow runtime changing particle size in openGL mode

### usability
- more runtime options vs compiletime options

### simulation managemant
- make the compute derivitives function more organized
- find better solution for the stuff in the algorithms file
- find better solution to switch options on and off
- find better solution for 2D simulation

### simulation features
- add functionality from the other branch
- linear consistency fix
- investigate angular momentum (not)conservation
- add collisional merging simulation

### integration
- better structure to different integrators
- mark deriviatives in the particle buffer
- adaptive algorithms

### performance
- somehow automatically adjust particle attributes based on options
- optimize the functions in models.h
- add datastructures

### output data handling
- use pinned host memory here
- remove the stream sync points

### library / utils
- add to string for cuda data types
- fix opengl frontend camera
- use logFlush for cuda assert
- fix cmake bug with multiple cuda gencode arguments

# finished
for motivation, all finished todo entries are moved here instead of beeing deleted

- resize callback for openGL
- fix particle numbers that are not power of 2
- write installation instructions

--- v0.2.0 ---
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
