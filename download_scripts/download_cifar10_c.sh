# Creates the data directory if it does not exist
if [ ! -d data ]; then
    mkdir data
fi

cd data
wget -O CIFAR-10-C.tar https://zenodo.org/records/2535967/files/CIFAR-10-C.tar?download=1
tar -xvf CIFAR-10-C.tar
rm -rf CIFAR-10-C.tar
cd ..