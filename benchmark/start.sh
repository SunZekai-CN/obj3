#!/bin/bash

# Activate python virtual env



# Distributed Training Metrics Generation --2 Processes --Testing CPU increase effect & Arch Difference #

# Feedforward Neural Net

#for (( i = 1; i < 6; i++ ))
#do
#    for (( j = 2; j <=20;j++ ))
#    do  
#        for ((k=1;k<10;k++))
#        do
#            python3 src/main.py --epochs=$i --nodes=1 --procs=$j --arch=ff --order=y
#        done 
#        for ((k=1;k<10;k++))
#        do
#           python3 src/main.py --epochs=$i --nodes=1 --procs=$j --arch=ff --order=n
#        done
#    done
#done

# Convolutional Neural Net
for (( i = 1; i < 6; i++ ))
do
    for (( j = 2; j <=20;j++ ))
    do  
        for ((k=1;k<10;k++))
        do
            python3 src/main.py --epochs=$i --nodes=1 --procs=$j --arch=conv --order=y
        done 
        for ((k=1;k<10;k++))
        do
           python3 src/main.py --epochs=$i --nodes=1 --procs=$j --arch=conv --order=n
        done
    done
done