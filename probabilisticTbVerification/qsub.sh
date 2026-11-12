#!/bin/bash
################################################################
#
## Example script to submit a PBS job for the probabilistic
#  verification of ensemble brightness temperature forecasts,
#  one job per channel.
#  Optional:
#  - maps / ensemble maps / histograms / per-case score plots
#
# Usage: ./qsub.sh
#
################################################################

root=$PWD

machine=`hostname`

case $machine in
    *"casper"* )
        account=NMMM0015
        queue=casper
        nodes=1
        ncpus=36
        mpiprocs=36
        mem=109GB
        walltime=01:00:00
        job_priority=economy
        ;;
    *"derecho"* )
        account=NMMM0015
        queue=main
        nodes=1
        ncpus=128
        mpiprocs=128
        mem=235GB
        walltime=01:00:00
        job_priority=economy
        ;;

    *)
        echo "${machine} not valid"
        exit 1
        ;;
esac
####################### VARIABLES SETUP ########################
# experiment directory (passed as -dir); member files are expected at
#   ${expDir}/Verification/fc/memNNN/YYYYMMDDHH/<fhr>hr/dbOut/obsout_hofx_${sensor}.h5
expDir=/glade/derecho/scratch/${USER}/pandac/EXPERIMENT_NAME  # edit
sensor=abi_g16          # abi_g16, ahi_himawari8
channels=(13)
dateIni=2018041500
dateEnd=2018051200
delta=72
firstHour=6
fcstHr=48
fcstFreq=6
thresholds="pct_5.0,pct_10.0,pct_15.0,pct_20.0,pct_25.0,val_235.0,val_245.0"
event=below             # event is Tb below (cloudy) or above the threshold
nBins=20
nReaders=64             # processes reading member files (I/O bound, can exceed ncpus)
maps=false
ensMaps=false           # one map per member, expensive
hist=false
cases=false

for ch in ${channels[@]}; do

    outputFolder=${sensor}_ch${ch}_${dateIni}-${dateEnd}_${fcstHr}hr
    mkdir -p ${outputFolder}
    cd ${outputFolder}

cat <<EOF > probVerif_${sensor}_ch${ch}.sh
#!/bin/bash
#PBS -N probVerif_${sensor}_ch${ch}
#PBS -l select=${nodes}:ncpus=${ncpus}:mpiprocs=${mpiprocs}:mem=${mem}
#PBS -l walltime=${walltime}
#PBS -q ${queue}
#PBS -A ${account}
#PBS -l job_priority=${job_priority}
#PBS -j oe
#PBS -o job.out

source /etc/profile.d/z00_modules.sh
module load conda/latest
conda activate npl
export PYTHONDONTWRITEBYTECODE=1

cd $PWD

echo "\`date\` STARTED DATE"
python ${root}/probabilisticVerification.py -dir ${expDir} -s ${sensor} -ch ${ch} \\
  -di ${dateIni} -de ${dateEnd} -dt ${delta} -fhr0 ${firstHour} -fhr ${fcstHr} -freq ${fcstFreq} \\
  -thres ${thresholds} -ev ${event} -nbins ${nBins} -nread ${nReaders} \\
  -maps ${maps} -ens ${ensMaps} -hist ${hist} -cases ${cases}
echo "\`date\` ENDED DATE"
exit 0
EOF

    chmod +x probVerif_${sensor}_ch${ch}.sh
    echo "Submitting job probVerif_${sensor}_ch${ch}.sh ..."
    qsub probVerif_${sensor}_ch${ch}.sh
    cd ..

done
exit 0
