#!/bin/bash

export FILEID=1pNngqXdc3cKr9uLNW-Hu3SKvOpjzfzGY
export FILENAME=Emotion_LLaMA.pth

wget --load-cookies /tmp/cookies.txt  \ 
"https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt \ 
--keep-session-cookies \ 
--no-check-certificate 'https://docs.google.com/uc?export=download&id=FILEID' -O-  \ 
| sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=FILEID" -O FILENAME 
