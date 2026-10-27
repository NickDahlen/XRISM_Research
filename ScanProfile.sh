#!/bin/bash -l
#SBATCH --time=26:00:00
#SBATCH --ntasks=2
#SBATCH --cpus-per-task=4
#SBATCH --mem=100G
#SBATCH --export=ALL

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"

echo "Starting job in: $(pwd)"

export CALDB="$HOME/xrism_caldb"
export CALDBCONFIG="$CALDB/caldb.config"
export CALDBALIAS="$CALDB/alias_config.fits"

source ~/miniforge3/etc/profile.d/conda.sh
conda activate henv

source "$HEADAS/headas-init.sh"

unset HEADASPROMPT
export HEADASNOQUERY=1
export HEADASLOGFILE="/tmp/heasoft_${SLURM_JOB_ID}.log"

export PFILES="/tmp/pfiles_${SLURM_JOB_ID}:$HEADAS/syspfiles"
mkdir -p "/tmp/pfiles_${SLURM_JOB_ID}"


obs_id1=$2
obs_id2=$3

python scanAllProfiles.py -f newStorageFull/$obs_id1 -s 2
python scanAllProfiles.py -f newStorageFull/$obs_id2 -s 2
