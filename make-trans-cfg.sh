
TEMP_DIR=scripts/trans_template/
CFG_DIR=scripts/trans_cfg/
VERSION=v7 

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

#ILS=(bel)
#RLS=(rus)

for i in ${!ILS[*]}; do
  IL=${ILS[$i]}
  RL=${RLS[$i]}
  echo $IL
  for f in $TEMP_DIR/bi-w-32000-cn $TEMP_DIR/bi-w-16000-cn; do
    sed "s/IL/$IL/g; s/RL/$RL/g; s/VERSION/$VERSION/g" < $f > ${f/trans_template/trans_cfg_$VERSION/}_$IL$RL.sh 
    chmod u+x ${f/trans_template/trans_cfg_$VERSION/}_$IL$RL.sh 
  done
done
