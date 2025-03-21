# Creates the data directory if it does not exist
if [ ! -d data ]; then
    mkdir data
fi

cd data
wget -O CIFAR-100-C.tar https://zenodo.org/records/3555552/files/CIFAR-100-C.tar?download=1
tar -xvf CIFAR-100-C.tar
rm -rf CIFAR-100-C.tar
cd ..