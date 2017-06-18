#This script requires gdrive commandline tool ro download dataset from google drive
#https://github.com/prasmussen/gdrive
URL=https://drive.google.com/uc?export=download&id=0B7EVK8r0v71pZjFTYXZWM3FlRnM
ZIP_FILE=./celeba.zip

echo "Downloading aligned and chopped celeba image dataset..."
#gdrive download $URL --path ./celeba.zip
wget --no-check-certificate https://drive.google.com/uc?export=download&id=0B7EVK8r0v71pZjFTYXZWM3FlRnM -O $ZIP_FILE