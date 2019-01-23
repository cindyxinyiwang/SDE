#!/bin/bash

TDIR=/projects/tir1/corpora/multiling-text/ted
UDIR=/projects/tir1/users/xinyiw1/utils
vocab_size=32000

#mkdir -p data/eng
#for split in train dev test; do
#  tail -n +2 $TDIR/__multialign/all_talks_$split.tsv | cut -f 1 | python src/reversible_tokenize.py  > data/eng/ted-$split.mtok.eng
#done

python $UDIR/train-spm.py \
  --input=data/eng/ted-train.mtok.eng \
  --model_prefix=data/eng/spm"$vocab_size".mtok.eng \
  --vocab_size="$vocab_size"


