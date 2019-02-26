# my todo list

## high priority

### bugfix / simulation management
- fix xsph

### usability
- print the settings used

### simulation feature
- balsara switch
- linear consistency fix and investigate angular momentum (not)conservation
- variable timestep

### testing / simulating
- find material properties
- find values for initial conditions
- run simulations?


## medium priority

### usability
- more runtime options vs compiletime options

### simulation management
- make the compute derivatives function more organized
- use some system to manage boundary conditions
- find better solution for 2D simulation

### integration
- automatically integrate what is needed using TMP

### performance
- somehow automatically adjust particle attributes based on options
- optimize the functions in models.h


## low priority / ideas / long term projects

### particle creation
- add python script support to generate initials

### particle management
- enhance compile time when using particle templates
- have attributes that are the same for all particles in constant memory
- better storage for long data types like matrices

### simulation management
- find better solution to switch options on and off

### integration
- better structure to different integrators
- adaptive algorithms

### performance
- check out shuffle operations an vote functions
- add datastructures
- try new graph style kernel launching
- MPI support
- share load between CPU and GPU
- better memory alignment for matrices

### simulation features
- add collisional merging simulation
- different materials (using recursive kernel launches)

### output data handling
- provide binary data format
- provide python script to generate (3d html) plots USE: ipyvolume



# finished
for motivation, all finished todo entries are moved here instead of being deleted

- fix bug in edot and rdot calculation
- add custom literal for floating point variables
- calculate some global constants at compile time
- use precomputed constants for kernel calls
- add the Clohessy-Wiltshire model
- add proper rng seeds for the initial conditions
- allow normalization of center of mass and velocity at center of mass
- add plummer model as a particle source including option for velocities in equilibrium
- reading from file can now be done for all combinations of particle attributes
- add new modifiers for the initial conditions
- rework initial condition generation to work better with the new particle storage system

--- v0.3.0 --- 17.01.2018
- option to use fast math computation to further enhance speed
- add possibility to store result data with different particle attributes
- improve performance of storing result data
- algorithms now use templates instead of macros
- fix initialisation function of particle buffer
- add function to concatenate and merge particles
- big overhaul of the particle storage system including:
    - easier to use correctly
    - better error messages if used incorrectly
    - load_particle without arguments now returns a particle with all attributes stored in the buffer, instead of an empty particle
    - host buffer, device buffer and device reference (previously device copy) have separate classes
    - device buffer can now load and store particles from host code
    - different particle buffers can be created from each other
    - pinned host memory can be used for host particle buffer
    - derivatives are now marked as such in the particle buffer
    - mapped and pinned state of resources is not overwritten after most assignments

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
