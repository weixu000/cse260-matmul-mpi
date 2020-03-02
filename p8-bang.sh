cd $SGE_O_WORKDIR

echo
echo " *** Current working directory"
pwd
echo
echo " *** Compiler"
# Output which  compiler are we using and the environment
mpicc -v
echo
echo " *** Environment"
printenv

echo

echo ">>> Job Starts"
date

# Commands go here

mpirun -np 1 ./apf -n 400 -i 2000 -x 1 -y 1
mpirun -np 2 ./apf -n 400 -i 2000 -x 1 -y 2
mpirun -np 4 ./apf -n 400 -i 2000 -x 1 -y 4
mpirun -np 8 ./apf -n 400 -i 2000 -x 1 -y 8
mpirun -np 2 ./apf -n 400 -i 2000 -x 1 -y 2 -k
mpirun -np 4 ./apf -n 400 -i 2000 -x 1 -y 4 -k
mpirun -np 8 ./apf -n 400 -i 2000 -x 1 -y 8 -k