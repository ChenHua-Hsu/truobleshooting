#!/bin/bash
WORKDIR=/afs/cern.ch/user/c/chenhua
cd ${WORKDIR}
source script/env.sh
cd /eos/home-c/chenhua/copy_tdsm_encoder_sweep16
/afs/cern.ch/user/c/chenhua/.local/bin/wandb agent calo_tNCSM/NCSM-encoder_sweep0605_configs_best_test/6ajv92w0 --count 1