#!/bin/bash
# auto_run.sh — HMC launcher for Tianhe
# Usage: ./auto_run.sh --nodes N [--Lz 16] [other hmc_run args...]
#
# Architecture (NUMA-optimal):
#   8 MPI ranks per node  — 1 rank per NUMA domain
#   8 OMP threads/rank    — 8 cores per NUMA domain
#   Lz must be divisible by (nodes * 8)
#
# Example: --nodes 2 --Lz 16 → 16 ranks, 2 nodes, Lz_local=1 per rank

set -e

CORES_PER_NODE=64
OMP=8              # 1 rank per NUMA domain (8 cores each)
RPN=$((CORES_PER_NODE / OMP))   # ranks per node = 8
PARTITION="thcp1"
TIME="24:00:00"
MPI_PATH="/usr/local/ompi/bin"

# --- Parse --nodes and --Lz/--Ls ---
ALL_ARGS=("$@")
NODES=""
LZ=16    # default matches params.hpp

for ((i=0; i<${#ALL_ARGS[@]}; i++)); do
    case "${ALL_ARGS[$i]}" in
        --nodes) NODES="${ALL_ARGS[$((i+1))]}" ;;
        --Lz)    LZ="${ALL_ARGS[$((i+1))]}" ;;
        --Ls)    LZ="${ALL_ARGS[$((i+1))]}" ;;
    esac
done

if [[ -z "$NODES" ]]; then
    echo "Error: --nodes is required"
    echo "Usage: ./auto_run.sh --nodes N [--Lx N] [--Ly N] [--Lz N] [--Lt N] ..."
    echo "  Lz must be divisible by nodes*${RPN} (= nodes * ranks/node)"
    exit 1
fi

NRANKS=$((NODES * RPN))   # total MPI ranks

if ((LZ % NRANKS != 0)); then
    echo "Error: Lz=${LZ} must be divisible by nodes*${RPN} = ${NRANKS}"
    echo "  Valid Lz for --nodes ${NODES}: $(python3 -c "print([x for x in range(1,256) if x%${NRANKS}==0])" 2>/dev/null || echo "multiples of ${NRANKS}")"
    exit 1
fi

LZ_LOCAL=$((LZ / NRANKS))
TOTAL_CORES=$((NODES * CORES_PER_NODE))

echo "============================================"
echo "  HMC Configuration"
echo "============================================"
echo "  Lattice Lz    = $LZ"
echo "  Nodes         = $NODES"
echo "  Ranks/node    = $RPN  (1 per NUMA domain)"
echo "  MPI ranks     = $NRANKS"
echo "  Lz_local      = $LZ_LOCAL  (z-slices per rank)"
echo "  OMP threads   = $OMP  (1 per core in NUMA domain)"
echo "  Total cores   = $TOTAL_CORES"
echo "============================================"

# --- Generate SLURM script ---
HMC_DIR="$(cd "$(dirname "$0")" && pwd)"
LOG_DIR="${HMC_DIR}/log"
SLURM_DIR="${HMC_DIR}/slurm"
mkdir -p "${SLURM_DIR}"
mkdir -p "${LOG_DIR}"
JOBSCRIPT=$(mktemp "${SLURM_DIR}/job_XXXXXX.slurm")

cat > "$JOBSCRIPT" << EOF
#!/bin/bash
#SBATCH --partition=${PARTITION}
#SBATCH --nodes=${NODES}
#SBATCH --ntasks-per-node=${RPN}
#SBATCH --cpus-per-task=${OMP}
#SBATCH --time=${TIME}
#SBATCH --job-name=hmc_n${NODES}
#SBATCH --output=${LOG_DIR}/hmc_n${NODES}_%j.log

export OMP_NUM_THREADS=${OMP}
export OMP_PROC_BIND=close
export OMP_PLACES=cores
export PATH=${MPI_PATH}:\$PATH

cd ${HMC_DIR}

srun ./build/hmc_run \\
    ${ALL_ARGS[@]} \\
    --omp ${OMP}
EOF

echo "Generated: $JOBSCRIPT"
echo ""
cat "$JOBSCRIPT"
echo ""

# --- Submit ---
sbatch "$JOBSCRIPT"
echo "Job submitted."
