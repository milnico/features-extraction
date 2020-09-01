#!/bin/bash

printUsage() {
    echo "---- USAGE OF BASH SCRIPT run ----"
    echo "This script must receive the following mandatory parameters:"
    echo "1) Python filename (i.e., run_es.py)"
    echo "2) Configuration file (.ini)"
    echo "An example of command is:"
    echo "./run run_es.py ErDpole.ini"
}

if [ $# -lt 2 ]
  then
    echo "ERROR! Invalid number of parameters!!! Check usage and retry!!!"
    printUsage
    exit -1
fi

# Importing configuration
. /srv/nodes/conf/nodeScripts.conf

CONTAINER="farsa"
PYFILE=$1
CONFIGFILE=$2
NUM_REPLICATIONS=1
if [ $# -ge 3 ]
  then
    NUM_REPLICATIONS=$3
fi
STARTING_SEED=1
if [ $# -ge 4 ]
  then
    STARTING_SEED=$4
fi
ALGO="Salimans"
if [ $# -ge 5 ]
  then
    ALGO=$5
fi
OUTDIR=$(pwd)
if [ $# -eq 6 ]
  then
    OUTDIR=$6
fi
FOLDER=$(pwd)
CONTAINERDIR=$FOLDER
CONTAINERDIR="$(echo $CONTAINERDIR | sed -r 's/\/home\/nicola/\//')"
CMD_EXE=python3.5

SEND_MAIL_OPTION=""
CONFIGURATION_FILE_DIRECTORY="$FOLDER"
SLURM_BATCH_FILE="${CONFIGURATION_FILE_DIRECTORY}/startBatch.sh"

cat > "$SLURM_BATCH_FILE" << EOF
#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --output="${CONFIGURATION_FILE_DIRECTORY}/${NODEOS_TOTAL99_OUTPUT_FILENAME}"
#SBATCH --nodelist=octo10
#SBATCH --mem=2gb


CUR_SEED="\$(( $STARTING_SEED + \$SLURM_ARRAY_TASK_ID ))"

# Arguments of cython
ARGS=" $CONTAINERDIR/$PYFILE -f $CONTAINERDIR/$CONFIGFILE -s \$CUR_SEED -a $ALGO -d $OUTDIR"

srun "${NODEOS_SCRIPTS_DIR}/cloneContainerAndStartInside.sh" "$CONTAINER" "\$(hostname)" "/" "$CMD_EXE" \$ARGS

EOF
errorExit $? $NODEOS_EC_CANNOT_GENERATE_BATCH_FILE_FOR_FARSA "ERROR: Cannot generate the batch file"
chmod 755 "$SLURM_BATCH_FILE"
errorExit $? $NODEOS_EC_CANNOT_GENERATE_BATCH_FILE_FOR_FARSA "ERROR: Cannot change permissions of the batch file"

sbatch --array="0-$(( $NUM_REPLICATIONS - 1 ))" ${SEND_MAIL_OPTION} "${SLURM_BATCH_FILE}"

echo
echo "Started $NUM_REPLICATIONS instance of ./run"
echo

exit $NODEOS_EXIT_SUCCESS


