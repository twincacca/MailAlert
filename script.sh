rm old-out* diff-*

# dd=$(date | sed 's/ /-/g')
dd=$(env TZ=Europe/Berlin date | sed 's/ /-/g')
#dd=$(env TZ=Europe/Berlin date +%Y%m%d  | sed 's/ /-/g')

#echo Ciaociao > out-$dd.txt
#date >> out-$dd.txt
#cat user_input.txt | cowsay >> out-$dd.txt

wget https://www.spalv.ch/it/animali/smarriti

