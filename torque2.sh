#!/bin/bash
#PBS -l nodes=1:ppn=1, mem=1gb,walltime=00:05:00
#PBS -M jd.prada11@uniandes.edu.co
#PBS -m abe
#PBS -N JuanPrada_Ejercicio27

# COMPILO CON chmod u+x torque2.sh Y EJECUTO CON ./torque2.sh 

module load mpich/3.2.1

mpicc integra.c -o integra -lm

for i in 1E2 1E4 1E5 1E6 1E7 1E8; 
do 
	mpiexec -n 1 ./integra 1000 > int"$i".txt 
done 

python plots.py 

rm *.txt



 

