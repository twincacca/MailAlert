rm out*
dd=$(date | sed 's/ /-/g')
echo Ciaociao > out-$dd.txt
date >> out-$dd.txt
cat user_input.txt | cowsay >> out-$dd.txt
