#!/bin/bash
#SBATCH --job-name="Milestone3"
#SBATCH --output="Milestone3.out"
#SBATCH --error="Milestone3.err"
#SBATCH --partition=gpuA40x4
#SBATCH --mem=12G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --constraint="projects"
#SBATCH --gpus-per-node=1
#SBATCH --gpu-bind=closest
#SBATCH --account=bche-delta-gpu
#SBATCH -t 00:03:00

module reset
module load nsys
module load ncu

echo -e "job $SLURM_JOBID is starting on `hostname`\n\n"

# srun ./m3 100 > m3_100.out
# srun ./m3 1000 > m3_1000.out
srun ./m3 10000 > m3_10000.out

# srun ./m3_op4  100 > m3_op4_100.out
# srun ./m3_op4  1000 > m3_op4_1000.out
# srun ./m3_op4  10000 > m3_op4_10000.out

# srun ./m3_op6  100 > m3_op6_100.out
# srun ./m3_op6  1000 > m3_op6_1000.out

# srun ./m3_req0 100 > m3_req0_100.out
# srun ./m3_req0 1000 > m3_req0_1000.out

# srun ./m3_req1 100 > m3_req1_100.out
# srun ./m3_req1 1000 > m3_req1_1000.out
# srun ./m3_req1 10000 > m3_req1_10000.out

# Generate nsys profile information, output will be in profile.out and report.nsys-rep
# srun nsys profile --stats=true ./m3 > profile.out

# Generate ncu profile information, output will be in analysis_file.ncu-rep
# srun ncu --set full -f -o analysis_file ./m3 > m3.out

# srun ncu --set full -f -o m3_ncu ./m3      10000 > m3_ncu.out

# srun ncu --set full -f -o op0_ncu ./m3_op0  10000 > op0_ncu.out
# srun ncu --set full -f -o op1_ncu ./m3_op1  10000 > op1_ncu.out
# srun ncu --set full -f -o op2_ncu ./m3_op2  10000 > op2_ncu.out
# srun ncu --set full -f -o op4_ncu ./m3_op4  10000 > op4_ncu.out
# srun ncu --set full -f -o op6_ncu ./m3_op6  10000 > op6_ncu.out

# srun ncu --set full -f -o req0_ncu ./m3_req0 10000 > req0_ncu.out
# srun ncu --set full -f -o req1_ncu ./m3_req1 10000 > req1_ncu.out

echo "All runs finished for job $SLURM_JOBID"