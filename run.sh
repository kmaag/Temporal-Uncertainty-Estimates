#!/bin/bash
#
#	usage: ./run.sh
#

clear 

# settings
export OPENBLAS_NUM_THREADS=1

# cython setup
python3 metrics_setup.py build_ext --inplace

# execute main python scripts
python3 main.py

printf "#-------------------------------#\n"
printf "# SCRIPTS EXECUTED SUCCESSFULLY #\n"
printf "#-------------------------------#\n"

