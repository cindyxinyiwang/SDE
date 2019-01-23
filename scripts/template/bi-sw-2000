#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --mem=12g
##SBATCH --nodelist=compute-0-7
#SBATCH -t 0

#export PYTHONPATH="$(pwd)"
#export CUDA_VISIBLE_DEVICES="2"

python src/main.py \
  --clean_mem_every 5 \
  --reset_output_dir \
  --output_dir="outputs_v7/bi-sw-2000_ILRL/" \
  --data_path data/ILRL_eng/ \
  --train_src_file_list data/ILRL_eng/ted-train.mtok.sepspm2000.ILRL \
  --train_trg_file_list  data/ILRL_eng/ted-train.mtok.spm8000.eng \
  --dev_src_file  data/IL_eng/ted-dev.mtok.spm2000.IL \
  --dev_trg_file  data/IL_eng/ted-dev.mtok.spm8000.eng \
  --dev_trg_ref  data/IL_eng/ted-dev.mtok.eng \
  --src_vocab_list  data/ILRL_eng/ted-train.mtok.sepspm2000.ILRL.vocab \
  --trg_vocab_list  data/ILRL_eng/ted-train.mtok.spm8000.eng.vocab \
  --d_word_vec=128 \
  --d_model=512 \
  --log_every=50 \
  --eval_every=2500 \
  --ppl_thresh=15 \
  --merge_bpe \
  --eval_bleu \
  --cuda \
  --batcher='word' \
  --batch_size 1500 \
  --valid_batch_size=7 \
  --patience 5 \
  --lr_dec 0.8 \
  --dropout 0.3 \
  --max_len 10000 \
  --seed 0
