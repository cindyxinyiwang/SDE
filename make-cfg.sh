
TEMP_DIR=scripts/template/
# change random seed and directory name as desired
CFG_DIR=cfg_s0/
VERSION=s0
SEED=0

mkdir -p scripts/"$CFG_DIR"
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
  for f in $TEMP_DIR/bi-w-64000 $TEMP_DIR/bi-sw-joint-32000 $TEMP_DIR/bi-sw-32000  $TEMP_DIR/bi-semb-bq-o32000 ; do
    sed "s/VERSION/$VERSION/g; s/SEED/$SEED/g; s/IL/$IL/g; s/RL/$RL/g" < $f > ${f/template/"$CFG_DIR"/}_$IL$RL.sh 
    chmod u+x ${f/template/"$CFG_DIR"/}_$IL$RL.sh 
  done
done
