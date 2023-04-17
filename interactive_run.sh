 srun --nodes=1 --ntasks-per-node=4 --gres=gpu:4  -N 1 -n 4 --partition=standard-g --account=project_465000356 ../select_gpu.sh ./mg
