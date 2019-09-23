#!/bin/bash

function show_help
{
    echo "simCreate, prepare parameter studies with GraSPH2."
    echo "Set a path, start and end number. Option -i will create all folders and copy the default setting files."
    echo "Option -c will compile GraSPH using the settings as found in the folder and save the executable file ready for simulation."
    echo "-s | --start        -  start number"
    echo "-e | --end          -  end number"
    echo "-p | --path         -  folder where simulations are build"
    echo "-i | --init-folders -  create folders and initialize with settings files"
    echo "-c | --configure    -  configure cmake"
    echo "-m | --make         -  make the executables"
    echo "-r | --run          -  add slurm jobs"
    echo "--clean             -  remove the cmake files build files, but keep the settings"
    echo "-g | --grasph       -  path to the GraSPH2 folder (only needed if you move the script)"
}



START_ID=0
END_ID=0
OUT_PATH=""
INIT=false
CONFIGURE=false
CLEAN=false
MAKE=false
RUNSLURM=false

GRASPH_PATH="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )/../"

SLURM_DEV=$(<${GRASPH_PATH}scripts/simCreate_defaultSLURM.txt)
CMAKE_DEV=$(<${GRASPH_PATH}scripts/simCreate_defaultCMAKE.txt)

if [[ ! $# -gt 0 ]] ; then
    show_help
    exit
fi

while [[ $# -gt 0 ]]
do
    key="$1"

    case $key in
        -s|--start)
        START_ID="$2"
        shift # past argument
        shift # past value
        ;;
        -e|--end)
        END_ID="$2"
        shift # past argument
        shift # past value
        ;;
        -p|--path)
        OUT_PATH="$2"
        shift # past argument
        shift # past value
        ;;
        -i|--init)
        INIT=true
        shift # past argument
        ;;
        -c|--configure)
        CONFIGURE=true
        shift # past argument
        ;;
        -m|--make)
        MAKE=true
        shift # past argument
        ;;
        -r|--run)
        RUNSLURM=true
        shift # past argument
        ;;
        --clean)
        CLEAN=true
        shift # past argument
        ;;
        -g|--grasph)
        GRASPH_PATH="$2"
        shift # past argument
        shift # past value
        ;;
        -h|--help)
        show_help
        shift # past argument
        ;;
        *)    # unknown option
        echo "unknown option $1"
        echo ""
        show_help
        exit
        ;;
    esac
done

if [ -z $OUT_PATH ] ; then
    echo "you need to provide a path to put your folders using the --path/-p option"
    exit
fi

# initialize folders as necessary
if [ "$INIT" = true ] ; then

    mkdir -p $OUT_PATH
    for i in `seq $START_ID $END_ID`
    do
        folder=$(printf "%s%03d" $OUT_PATH $i)
        echo "create and initialize $folder"
        mkdir $folder
        mkdir ${folder}/build
        cp ${GRASPH_PATH}settings ${folder}/ -r
        echo "${CMAKE_DEV}" > ${folder}/cmakeOptions.txt
        echo "${SLURM_DEV}" > ${folder}/sbatchJobfile.cmd
        echo "srun ${folder}/build/GraSPH2 > ${folder}/prog.out" >> ${folder}/sbatchJobfile.cmd
    done
fi

# clean up cmake config and build files if desired
if [ "$CLEAN" = true ] ; then

    for i in `seq $START_ID $END_ID`
    do
        folder=$(printf "%s%03d" $OUT_PATH $i)
        echo "remove build files from ${folder}/build/"
        rm ${folder}/build/* -r
    done
fi

# configure if desired
if [ "$CONFIGURE" = true ] ; then

    for i in `seq $START_ID $END_ID`
    do
        folder=$(printf "%s%03d" $OUT_PATH $i)
        echo "use cmake to configure $folder"
        cmoptions=$(<${folder}/cmakeOptions.txt)
        cmake -B${folder}/build -H${GRASPH_PATH} -DCUSTOM_SETTING_PATH=${folder}/settings $cmoptions
    done
fi

# make if desired
if [ "$MAKE" = true ] ; then

    for i in `seq $START_ID $END_ID`
    do
        folder=$(printf "%s%03d" $OUT_PATH $i)
        echo "build simulation in $folder"
        make -C ${folder}/build/ -j2
    done
fi

# add to slurm if desired
if [ "$RUNSLURM" = true ] ; then

    for i in `seq $START_ID $END_ID`
    do
        folder=$(printf "%s%03d" $OUT_PATH $i)
        echo "start slurm job in $folder"
        cd $folder
        sbatch sbatchJobfile.cmd
    done
fi