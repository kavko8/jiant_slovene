#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1
#SBATCH --time=48:00:00
#SBATCH --mem=40GB

module purge

JIANT_PATH=$(pwd)/jiant/jiant
echo $MODELS_DIR
echo $MODEL_TYPE
echo $RUN_CONFIG_DIR
echo $SEED
export PYTHONPATH=jiant/

SHORT_MODEL_NAME="${MODEL_TYPE##*/}"

python $JIANT_PATH/proj/main/runscript.py   \
    run_with_continue \
       --jiant_task_container_config_path ${RUN_CONFIG_DIR} \
       --hf_pretrained_model_name_or_path ${MODEL_TYPE} \
       --model_path ${MODELS_DIR}/${SHORT_MODEL_NAME}/model/model.p \
       --model_config_path ${MODELS_DIR}/${SHORT_MODEL_NAME}/model/config.json \
       --model_load_mode from_transformers \
       --learning_rate $LR \
       --force_overwrite \
       --do_train \
       --do_val \
       --do_save \
       --eval_every_steps $VAL_INTERVAL \
       --no_improvements_for_n_evals 30 \
       --save_checkpoint_every_steps $VAL_INTERVAL \
       --output_dir ${OUTPUT_DIR}/${TASK_NAME}/config_${CONFIG_NO} \
       --seed $SEED \
       --save_model_every_logscale \
       --save_every_steps 5000
