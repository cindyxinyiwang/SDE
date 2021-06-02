import sys

col = int(sys.argv[1])

for line in sys.stdin:
  cols = line.strip().split(' ||| ')
  if len(cols) < 2: continue
  if col == 1 and cols[col] == '#untranslated':
    print(cols[0])
  else: 
    print(cols[col])
