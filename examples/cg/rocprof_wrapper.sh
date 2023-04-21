#!/bin/bash
set -euo pipefail
name="$1"
if [[ -n ${OMPI_COMM_WORLD_RANK+z} ]]; then
   export MPI_RANK=${OMPI_COMM_WORLD_RANK}
elif [[ -n ${MV2_COMM_WORLD_RANK+z} ]]; then
  export MPI_RANK=${MV2_COMM_WORLD_RANK}
elif [[ -n ${SLURM_PROCID+z} ]]; then
  export MPI_RANK=${SLURM_PROCID}
else
  echo "Unknown MPI layer detected! Must use OpenMPI , MVAPICH, or Slurm"
  exit 1
fi

export ROCR_VISIBLE_DEVICES=$SLURM_LOCALID

rocprof="/opt/rocm/bin/rocprof"
pid="$$"
outdir="rank_${pid}_${MPI_RANK}"
outfile="${name}_${pid}_${MPI_RANK}.csv"
${rocprof} -d ${outdir} -o ${outdir}/${outfile} --hsa-trace --hip-trace --roctx-trace "${@:2}"
