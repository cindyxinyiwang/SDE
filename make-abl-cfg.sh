

TEMP_DIR=scripts/template/
CFG_DIR=cfg_v7_abl_s1/
VERSION=v7_abl_s1
SEED=1

mkdir -p scripts/"$CFG_DIR"
#ILS=(
#  aze)
#RLS=(
#  tur)
ILS=(
  aze
  bel
  glg
  slk)
RLS=(
  tur
  rus
  por
  ces)

for i in ${!ILS[*]}; do
  IL=${ILS[$i]}
  RL=${RLS[$i]}
  echo $IL
  for f in $TEMP_DIR/bi-semb-bq-o8000-no-spe $TEMP_DIR/bi-semb-bq-o8000-no-w $TEMP_DIR/bi-semb-bq-o8000-no-char $TEMP_DIR/bi-semb-bq-sw-8000 $TEMP_DIR/bi-semb-bpe-o8000; do
    sed "s/VERSION/$VERSION/g; s/SEED/$SEED/g; s/IL/$IL/g; s/RL/$RL/g" < $f > ${f/template/"$CFG_DIR"/}_$IL$RL.sh 
    chmod u+x ${f/template/"$CFG_DIR"/}_$IL$RL.sh 
  done
done
