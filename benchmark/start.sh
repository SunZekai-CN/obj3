#!/bin/bash

# Activate python virtual env



# Distributed Training Metrics Generation --2 Processes --Testing CPU increase effect & Arch Difference #

# Feedforward Neural Net
timeout=(1.0 0.1 0.01 0.01 0.01 0.01 0.01 0.001 0.001 0.001 0.001 0.001 0.001)
workers=(20 20 11 12 14 16 18 10 11 12 13 15 18)
for ((i=0;i<13;i++))
do
    for ((j=1;j<=6;j++))
    do
        for ((k=0;k<20;k++))
        do
            python3 main.py --epochs=$j --workers=${workers[i]} --arch=ff --timeout=${timeout[i]}
        done
    done
    
done
# Convolutional Neural Net
#for (( i = 1; i < 6; i++ ))
#do
#    for (( j = 2; j <=20;j++ ))
#    do  
#        for ((k=1;k<10;k++))
#        do
#            python3 src/main.py --epochs=$i --nodes=1 --procs=$j --arch=conv --order=y
#        done 
#        for ((k=1;k<10;k++))
#        do
#           python3 src/main.py --epochs=$i --nodes=1 --procs=$j --arch=conv --order=n
#        done
#    done
#done