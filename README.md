# GraSPH2

A flexible c++ code for physical particle simulations.
GraSPH2 uses the CUDA API to do most of it's calculations on a GPU,
achieving speeds much higher than traditional CPU simulations.
Developed for applications in astrophysics, GraSPH2 can be used to
perform pure n-body simulations. It's main selling point however is the
integration of SPH which allows GraSPH2 to simulate fluids as well as
deformable solid body and granular materials.

Simulations can also be run either in  "precision mode" or in real time,
featuring an interactive openGL visualization.

---

## installation

To use GraSPH you have to compile it yourself from sources.

### dependencies

To compile GraSPH2 yourself you will need:

- CUDA 9
- gcc 6.4
- cmake 3.8 or higher
- [mpUtils v0.11.1](http://www.github.com/hschwane/mpUtils)

If you want to use real time simulation / visualization you will additionally need:

- glm
- glfw3
- glew
- graphics drivers supporting some OpenGL version (preferably 4.5+)
- make sure mpUtils was compiled with graphics features enabled

After installing the third party dependencies, go to the
[github page of mpUtils](http://www.github.com/hschwane/mpUtils) and follow
the installation instructions there.

###### HINT
If you happen to run on the "Cobra" HPC system at MPCDF, there is a script `scripts/loadCobraModules.sh` that will load all dependencies
to run and install mpUtils and GraSPH2. Consider adding `/path/to/this/scrip/loadCobraModules.sh` to your bashrc. 

### build

After that, use the following commands to download and install the newest
version of GraSPH2.

```
git clone https://github.com/hschwane/GraSPH2.git
cd GraSPH2
mkdir build
cd bin
cmake ..
make -j 8 #<cahnge 8 to the number of procesors you want to use for compiling>
```

After this the GraSPH2 executable can be found in th GraSPH2/build folder.

###### HINT
If mpUtils was installed in a non standard path remember to set the
`CMAKE_PREFIX_PATH` or `mpUtills_DIR` cmake variable.

### cmake options

The following additional cmake options are available:

- `OPENGL_FRONTEND` this option will only be availible if openGL was found on your system during theinstallation
                        of mpUtils. Turn this on to have your simulations drawn on screen in real time.
- `FORCE_NEW_VERSION` if enabled a instead of using the version number of the git commit, the version is increased by one.
                        This is mainly used during development and testing.
- `USE_FAST_MATH` enable unsafe math optimizations in device code. Better performance, but might impact accuracy.
- `USE_DEBUG_LOGGING` compile debug logging commands even in release builds.


## usage

### headless mode

To start a simulation just launch the executable file. Progress information will be written
to the console regularly and the program will terminate once the set time limit is reached.

### interactive mode

After launching the executable a window will open showing
the initial conditions. You can load also load a tsv or hdf5 file by dragging it onto the window.
Press "1" on your keyboard to start the simulation. You can pause it again at any time using the "2"
key. The Camera can be moved using "W", "A", "S" and "D", like in a video  game.
To fly up and down use "Q" and "E". "+" and "-" can be used to change the cameras movement speed.
Hold the left mouse button to look around. Use "R" to switch the camera rotation to a "trackball" style mode.
In Trackball mode you can Zoom using your mouse wheel and pan by holding the middle mouse button. 
The console will display some timing information.
Press "F" to switch between different color visualization modes.
Keys "T" and "G" control the size of the rendered particles.
Pressing "V" will show a UI-Window containing display settings for the visualization

### settings

GraSPH2 makes heavy use of templates to be as efficient as possible.
Therefore most of the simulation settings need to be done at compile time.
Options can be found in the folder `src/settings` and are split among several files.

### initial conditions

Initial conditions can be load from a file or be generated at the start of the simulation.
See `src/settings/settings.h` for all options. You can also extend the main() function  in
`main.cu` to generate different initial conditions.

### results

In the file `src/settings/outputSettings.h` you can name a folder where simulation results are written.
For each step of the simulation a new file is generated. 
You can use .tsv or hdf5 files if libhdf5 is availibe on your system.

## bugs / missing features / contributing

GraSPH2 is a quite young project and actively developed only by myself and only part time.
If you have any issues using the code, find any bugs or are in need of some
special feature not availible yet don't hesitate to contact me via mail
or create a ticket on github.

You may also try to extend / modify the code yourself if you know any c++.
CUDA knowledge is only required if you want to modify the deeper infrastructure
of the code, not for extending the physical models.


-----------------------
## included 3rd party code

[HighFive - Header-only C++ HDF5 interface](https://github.com/BlueBrain/HighFive)
