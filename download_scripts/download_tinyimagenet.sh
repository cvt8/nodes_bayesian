# Creates the data directory if it does not exist
if [ ! -d data ]; then
    mkdir data
fi

cd data
wget -O TinyImageNet.zip http://cs231n.stanford.edu/tiny-imagenet-200.zip
unzip TinyImageNet.zip
rm -rf TinyImageNet.zip
mv tiny-imagenet-200 TinyImageNet
cd ..