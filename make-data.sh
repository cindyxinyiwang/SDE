#!/bin/bash

#### change the directory to the directory where you put SDE repo
BASE=/projects/tir1/users/xinyiw1/
XDIR=$BASE/SDE
UDIR=$BASE/SDE/src

#### change the directory to where you store the data files
DDIR=/projects/tir1/corpora/multiling-text


### change the vocab size as you wish 
vocab_size=32000

DATA_PRE=ted
# low-resource language codes
ILS=(
  aze
  bel
  glg
  slk)
# paired high-resource language codes
RLS=(
  tur
  rus
  por
  ces)

for i in ${!ILS[*]}; do
  IL=${ILS[$i]}
  RL=${RLS[$i]}
  echo $IL

  mkdir -p  data/"$IL"_eng
  mkdir -p  data/"$RL"_eng
  mkdir -p  data/"$IL$RL"_eng
  
  ln -s $DDIR/ted/"$IL"_eng/ted-{train,dev,test}.orig."$IL"-eng data/"$IL"_eng/
  ln -s $DDIR/ted/"$RL"_eng/ted-{train,dev,test}.orig."$RL"-eng data/"$RL"_eng/
  
  for f in data/"$IL"_eng/*.orig.*-eng  data/"$RL"_eng/*.orig.*-eng; do
    src=`echo $f | sed 's/-eng$//g'`
    trg=`echo $f | sed 's/\.[^\.]*$/.eng/g'`
    echo "src=$src, trg=$trg"
    python cut-corpus.py 0 < $f > $src
    python cut-corpus.py 1 < $f > $trg
  done
  
  for f in data/"$IL"_eng/*.orig.{eng,$IL}  data/"$RL"_eng/*.orig.{eng,$RL}; do
    f1=${f/orig/mtok}
    #cat $f | perl $MDIR/scripts/tokenizer/tokenizer.perl > $f1
    cat $f | python src/reversible_tokenize.py > $f1
  done
  
  for split in train; do
    cat data/"$IL"_eng/$DATA_PRE-$split.orig."$IL" data/"$RL"_eng/$DATA_PRE-$split.orig."$RL"  > data/"$IL$RL"_eng/$DATA_PRE-$split.orig.$IL$RL
    cat data/"$IL"_eng/$DATA_PRE-$split.mtok."$IL" data/"$RL"_eng/$DATA_PRE-$split.mtok."$RL"  > data/"$IL$RL"_eng/$DATA_PRE-$split.mtok.$IL$RL
    cat data/"$IL"_eng/$DATA_PRE-$split.orig.eng data/"$RL"_eng/$DATA_PRE-$split.orig.eng  > data/"$IL$RL"_eng/$DATA_PRE-$split.orig.eng
    cat data/"$IL"_eng/$DATA_PRE-$split.mtok.eng data/"$RL"_eng/$DATA_PRE-$split.mtok.eng  > data/"$IL$RL"_eng/$DATA_PRE-$split.mtok.eng
  done
  
  for split in dev test; do
    cp data/"$IL"_eng/$DATA_PRE-$split.orig.$IL data/"$IL$RL"_eng/
    cp data/"$IL"_eng/$DATA_PRE-$split.mtok.$IL data/"$IL$RL"_eng/
    cp data/"$IL"_eng/$DATA_PRE-$split.orig.eng data/"$IL$RL"_eng/
    cp data/"$IL"_eng/$DATA_PRE-$split.mtok.eng data/"$IL$RL"_eng/
    cp data/"$RL"_eng/$DATA_PRE-$split.orig.$RL data/"$IL$RL"_eng/
    cp data/"$RL"_eng/$DATA_PRE-$split.mtok.$RL data/"$IL$RL"_eng/
    cp data/"$RL"_eng/$DATA_PRE-$split.orig.eng data/"$IL$RL"_eng/
    cp data/"$RL"_eng/$DATA_PRE-$split.mtok.eng data/"$IL$RL"_eng/
  done
  
  echo "train spm from data/'$IL$RL'_eng/$DATA_PRE-train.mtok.'$IL$RL'"
  python $UDIR/train-spm.py \
    --input=data/"$IL$RL"_eng/$DATA_PRE-train.mtok."$IL$RL" \
    --model_prefix=data/"$IL$RL"_eng/spm"$vocab_size.mtok.$IL$RL" \
    --vocab_size="$vocab_size" 
  
  echo "train spm from data/'$IL'_eng/'$DATA_PRE'-train.mtok.'$IL'"
  python $UDIR/train-spm.py \
    --input=data/"$IL"_eng/"$DATA_PRE"-train.mtok."$IL" \
    --model_prefix=data/"$IL"_eng/spm"$vocab_size.mtok.$IL" \
    --vocab_size="$vocab_size"
  
  echo "train spm from data/'$RL'_eng/'$DATA_PRE'-train.mtok.'$RL'"
  python $UDIR/train-spm.py \
    --input=data/"$RL"_eng/"$DATA_PRE"-train.mtok."$RL" \
    --model_prefix=data/"$RL"_eng/spm"$vocab_size.mtok.$RL" \
    --vocab_size="$vocab_size"
  
  for f in data/"$IL"_eng/*.mtok.eng data/"$RL"_eng/*.mtok.eng data/"$IL$RL"_eng/*.mtok.eng; 
  do
    python $UDIR/run-spm.py \
      --model=data/eng/spm"$vocab_size".mtok.eng.model \
      < $f \
      > ${f/mtok/mtok.spm$vocab_size} 
  done
  
  for f in data/"$IL"_eng/*.mtok."$IL"; 
  do
    python $UDIR/run-spm.py \
      --model=data/"$IL"_eng/spm"$vocab_size.mtok.$IL".model \
      < $f \
      > ${f/mtok/mtok.spm$vocab_size} 
  done
  
  for f in data/"$RL"_eng/*.mtok."$RL"; 
  do
    python $UDIR/run-spm.py \
      --model=data/"$RL"_eng/spm"$vocab_size.mtok.$RL".model \
      < $f \
      > ${f/mtok/mtok.spm$vocab_size} 
  done
  
  for f in data/"$IL$RL"_eng/*.mtok.{"$IL$RL","$IL"}; 
  do
    python $UDIR/run-spm.py \
      --model=data/"$IL$RL"_eng/spm"$vocab_size.mtok.$IL$RL".model \
      < $f \
      > ${f/mtok/mtok.spm$vocab_size} 
  done
  
  cat data/"$IL"_eng/$DATA_PRE-train.mtok.spm$vocab_size."$IL" data/"$RL"_eng/$DATA_PRE-train.mtok.spm$vocab_size."$RL"  > data/"$IL$RL"_eng/$DATA_PRE-train.mtok.sepspm$vocab_size.$IL$RL
  
  for f in data/"$IL"_eng/*train.mtok.spm$vocab_size.* data/"$RL"_eng/*train*.mtok.spm$vocab_size.* data/"$IL$RL"_eng/$DATA_PRE-train.mtok.spm$vocab_size.{eng,$IL$RL} data/"$IL$RL"_eng/$DATA_PRE-train.mtok.sepspm$vocab_size.$IL$RL; do
    echo "python $XDIR/src/get_vocab.py < $f > $f.vocab &"
    python $XDIR/src/get_vocab.py < $f > $f.vocab &
  done

  for f in data/"$IL"_eng/ted-train.mtok.spm$vocab_size."$IL" data/"$RL"_eng/ted-train.mtok.spm$vocab_size."$RL" data/"$IL$RL"_eng/ted-train.mtok.spm$vocab_size."$IL$RL"; do
    echo "python $XDIR/src/get_char_vocab.py < $f > $f.char5vocab &"
    python $XDIR/src/get_char_vocab.py --n 5 < $f > $f.char5vocab &
    echo "python $XDIR/src/get_char_vocab.py --orderd --n 5 < $f > $f.ochar5vocab &"
    python $XDIR/src/get_char_vocab.py --ordered --n 5 < $f > $f.ochar5vocab &
  done
 
  wait
done
