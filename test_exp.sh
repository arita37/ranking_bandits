#!/usr/bin/env bash

source activate bandit

TESTDIR=`pwd`/test
DATA_FNAME=`pwd`/hist.csv
mkdir $TESTDIR
mkdir $TESTDIR/dev_null
mkdir $TESTDIR/dev_null_bis

set -x
python exp.py --play 2 100 -r 10 --csv $DATA_FNAME --TopRank 100 --horizon_time_known $TESTDIR/dev_null --force || exit
python exp.py --merge 100 -r 10 --csv $DATA_FNAME --TopRank 100 --horizon_time_known $TESTDIR/dev_null || exit

