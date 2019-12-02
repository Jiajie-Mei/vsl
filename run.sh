#!/usr/bin/env bash


source /home/LAB/meijj/Env/miniconda3/etc/profile.d/conda.sh
conda activate vsl
export PATH="/home/LAB/meijj/Env/cudnn-10.1-v7.6.3:/usr/local/cuda-10.0/bin:/usr/local/cuda-10.1/bin:/usr/local/cuda-10.1/NsightCompute-2019.1${PATH:+:${PATH}}"
export LD_LIBRARY_PATH="/home/LAB/meijj/Env/cudnn-10.1-v7.6.3/lib64:/usr/local/cuda-10.0/lib64:/usr/local/cuda-10.1/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"

user_home=/home/LAB/wangcy


dataset=${1}
model="${2}"
[[ "${model}" == "" ]] && exit 1

[[ ${model} == f ]] && model=flat
[[ ${model} == h ]] && model=hier

num_unlabeled="${3}"
[[ "${num_unlabeled}" == "" ]] && num_unlabeled=0
tagging="${4}"
[[ "${tagging}" == "" ]] && tagging=iobes


data_root=${user_home}/Datasets/semi_supervised
data_home=${data_root}/Raw

use_unlabeled=False
[[ ${num_unlabeled} -ne 0 ]] && use_unlabeled=True


raw_dataset=CoNLL2003
[[ ${dataset} == conll2000 ]] && raw_dataset=CoNLL2000
[[ ${tagging} == iobes ]] && raw_dataset=${raw_dataset}_iobes

if [[ ${dataset} == conll2003 ]]; then
  srun -p cpu python -u process_ner_data.py \
  --train ${data_home}/${raw_dataset}/eng.train \
  --dev ${data_home}/${raw_dataset}/eng.testa \
  --test ${data_home}/${raw_dataset}/eng.testb \
  --unlabeled ${data_home}/${raw_dataset}/one_billion_words_$(printf "%06d" ${num_unlabeled}).txt
elif [[ ${dataset} == conll2000 ]]; then
  srun -p cpu python -u process_ner_data.py \
  --train ${data_home}/${raw_dataset}/split_train.txt \
  --dev ${data_home}/${raw_dataset}/split_dev.txt \
  --test ${data_home}/${raw_dataset}/test.txt \
  --unlabeled ${data_home}/${raw_dataset}/one_billion_words_$(printf "%06d" ${num_unlabeled}).txt
fi


main_file=vsl_gg.py
[[ ${model} == g ]] && main_file=vsl_g.py


srun --gres=gpu:V100:1 python -u ${main_file} \
--prefix ./log \
--model ${model} \
--dataset ${dataset} \
--prior_file prior.bin \
--vocab_file vocab \
--data_file ner.data \
--use_unlabel=${use_unlabeled} \
--unlabel_file ner_unlabel.data \
--embed_file ${data_root}/pretrained_embeddings/glove.6B.100d.txt \
--embed_type glove \
--tag_file ner_tagfile > log_train.txt 2>&1

