# my todo list

## high priority

### simulation feature
- add variable smoothing lenght
- figure out new eos based on molecular physics
- figure out stress tensor for new  model

## medium priority

### bugfixes
- fix xsph

### simulation feature
- linear consistency fix and investigate angular momentum (not)conservation

### simulation management
- make the compute derivatives function more organized
- find better solution for 2D simulation
- make eos and kernel selectable

### integration
- better structure to different integrators
- test different integrators
- automatically integrate what is needed using TMP

### frontends
- update frontend to new mputils version

## low priority / ideas / long term projects

### usability
- more runtime options vs compiletime options

### particle creation
- add python script support to generate initials
- just in time compiler to allow runtime settings changes

### particle management
- add cpu host reference
- enhance compile time when using particle templates
- have attributes that are the same for all particles in constant memory
- better storage for long data types like matrices
- better storage for vec3 / enable cpu vectorization

### simulation management
- find better solution to switch options on and off
- overhaul the frontend system 
- put simulation and integration in apropriate submodules
- add more interactive features
- use some system to manage boundary / initial conditions

### initial conditions
- add old initial condition generator (eg 2d things usw)
- add python support for initial conditions

### performance
- split work among multiple threads for "for each pair"
- profile and maybe optimize some functions 
- check out shuffle operations and vote functions
- add datastructures
- try new graph style kernel launching
- MPI support
- share load between CPU and GPU
- better memory alignment for matrices
- somehow automatically adjust particle attributes based on options
- fix the timestep download

### simulation features
- some kind of particle merging
- different materials (using recursive kernel launches)
- internal energy
- pcisph or similar
- surface forces for water
- proper boundary conditions

### output data handling
- provide python script to generate (3d html) plots USE: ipyvolume

### visualization
- allow playback of old simulations
- volume renderer
- ssao / better depth hints
- better transfer function
- more interactivity eg show parameters of particle etc

# finished
for motivation, all finished todo entries are moved here instead of being deleted

--- v0.7.1 --- 20.12.2019
- use new mpUtils version  
add file to load modules on cobra for easy installation

--- v0.7.0 --- 21.10.2019
- add simple UI for graphics settings
- make compatible with new mpUtils
- add option to keep particles in a simple box boundary  
- add option for constant acceleration (like gravity)
- clean out main a bit and add more structure to the simulation code
- add option to turn off solids extension (deviatoric stress)
- add density summation
- add variable timestep leapfrog 

--- v0.6.0 --- 23.09.2019
- better error message for simCreate script
- add edge highlighting to visualization
- better opengl visualization with real looking spheres
- change output filenames for windows

--- v0.5.1 --- 11.08.2019
- fix bugs on hdf5 loading and writing

--- v0.5.0 --- 11.08.2019
- update slurm scripts to work on new cobra gpus
- allow file loading via drag and drop on the openGL frontend
- add hdf5 support for reading files
- add hdf5 support for storing files

--- v0.4.0 --- 27.04.2019
- add batch script to run simulations with different settings
- add possibility to specify a custom location for settings
- fix openGL frontend for double precision runs
- fix division by zero in plasticity calculation
- add variable cutoff to plummer distribution
- print the settings used to a output file
- add logfile to output files
- better naming of output files
- add balsara switch
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

--- v0.3.0 --- 17.01.2019
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
