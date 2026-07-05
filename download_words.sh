#!/usr/bin/env bash
# Fetch a real Kazakh frequency word list and extract clean Cyrillic words.
set -e
curl -sL "https://raw.githubusercontent.com/hermitdave/FrequencyWords/master/content/2016/kk/kk_50k.txt" -o kk_50k.txt
python3 - << 'PY'
KZ=set('абвгдеёжзийклмнопрстуфхцчшщъыьэюяіңғүұқөһ')
words=[]
for l in open('kk_50k.txt',encoding='utf-8').read().splitlines():
    w=l.split(' ')[0].lower()
    if w and all(c in KZ for c in w) and 2<=len(w)<=12: words.append(w)
open('kk_words.txt','w',encoding='utf-8').write('\n'.join(words))
print('wrote kk_words.txt',len(words),'words')
PY
