rm old-out* diff-*

# dd=$(date | sed 's/ /-/g')
dd=$(env TZ=Europe/Berlin date | sed 's/ /-/g')
#dd=$(env TZ=Europe/Berlin date +%Y%m%d  | sed 's/ /-/g')

#echo Ciaociao > out-$dd.txt
#date >> out-$dd.txt
#cat user_input.txt | cowsay >> out-$dd.txt

# check cani/gatti smarriti in zona
#rename 's/out-cani-gatti-/old-out-cani-gatti-/' * # non va piu' rename, allora farne uno io:
#bash rename.sh
#-------- rename out-cani-gatti-* ---------
ls out-cani-gatti-* | awk '{print "mv " $1 " old-"$1}' > run.sh
bash run.sh
rm run.sh
#-------- rename out-cani-gatti-* ---------END

wget https://www.spalv.ch/it/animali
egrep -i "muralto|minusio|tenero|locarno" animali* --color | sed 's/ /\n/g' | grep href | grep -v "><img" > out-cani-gatti-$dd.txt
rm animali*
diff old-out-cani-gatti-* out-cani-gatti-* > diff-cani-gatti-$dd.txt
#rm -r out-cani-gatti-*

