# my todo list

## high priority

- add possibility to dump simulation to file
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
- better options for both frontends

### usability
- write installation instructions
- add some real time control and feedback to the frontends

### simulation managemant
- make the compute derivitives more organized
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