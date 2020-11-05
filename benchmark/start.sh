#!/bin/bash

# Activate python virtual env



# Distributed Training Metrics Generation --2 Processes --Testing CPU increase effect & Arch Difference #

# Feedforward Neural Net
timeout=100.0
for ((i=1;i<=7;i++))
do
    timeout=`echo "scale=5; $timeout/10.0" | bc`
    for (( j = 1; j <=3;j++ ))
    do  
        for ((k=1;k<=20;k++))
        do
            for ((m=1;m<=20;m++))
            do
                python3 main.py --epochs=$j --workers=$k --arch=ff --timeout=$timeout
            done
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