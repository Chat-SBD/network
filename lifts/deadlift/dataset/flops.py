from mpi4py import MPI
from sys import argv
from time import time

# FLOPS: Floating point operations per second
myops = 0

# for the seconds specified...
start = time()
seconds = float(argv[1])
while time() - start < seconds:
    # perform 2 floating point operations and record to counter
    mult = 3758.43789 * 37285.47382
    div = 1783.378194 / 314873.327853
    myops += 2

# my rank out of a number of initiated processes
comm = MPI.COMM_WORLD
rank = comm.Get_rank()

print(f"Process {rank} on {MPI.Get_processor_name()} calculated {myops} operations in {seconds} seconds")

# sum operations from all processes
operations = comm.reduce(myops, op = MPI.SUM, root = 0)

if rank == 0:
    flops = operations / seconds

    print("----------------------------------------------------------------")
    print(f"Calculated a total of {operations} operations in {seconds} seconds...")
    print(f"Final stats: | {round(flops, 5)} FLOPS | {round(flops / 1000000, 5)} GigaFLOPS | {round(flops / 1000000000, 5)} TeraFLOPS |")
    print("----------------------------------------------------------------")