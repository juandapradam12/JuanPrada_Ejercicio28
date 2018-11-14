#!/bin/bash
#PBS -l nodes=1:ppn=1, mem=1gb,walltime=00:05:00
#PBS -M jd.prada11@uniandes.edu.co
#PBS -m abe
#PBS -N JuanPrada_Ejercicio27

# COMPILO CON chmod u+x torque.sh Y EJECUTO CON ./torque.sh 

module load mpich/3.2.1

cd /hpcfs/home/fisi4028/jd.prada11/JuanPrada_Ejercicio28/

make clean
make Listing12.7
