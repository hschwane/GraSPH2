# my todo list

## high priority

- make all settings from the settings file actually work

## medium priority

### bugfixes
- fix particle numbers that are not power of 2
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
- resize callback for openGL
- fix opengl frontend camera

### usability
- write installation instructions
- more runtime options vs compiletime options

### simulation managemant
- make the compute derivitives function more organized
- add some kind of structure to the deriviative computation
- find better solution for the stuff in the algorithms file

### simulation features
- add functionality from the other branch
- linear consistency fix
- investigate angular momentum (not)conservation
- add collisional merging simulation

### output data handling
- use pinned host memory here
- remove the stream sync points

# finished
for personal motivation, all finished todo entries are moved here instead of beeing deleted

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
