# Dissertation

## Prepare machine

1. Go to https://console.cloud.google.com/compute/instances
2. If you are entering first time you need to sign up and create project
3. Click "CREATE INSTANCE"
4. Fill name, choose or leave region + zone
5. Choose GPUs, GPU type: NVIDIA Tesla P4 and Machine type: n1-standard-2
6. Click "Switch image" and choose Version: "Debian 10 based Deep Learning VM for TensorFlow Enterprise 2.9 with CUDA 11.3 M109"
7. In "Advanced options" click "ADD NEW DISK". We want to use separate disk to upload our files at once and then attach this disk to our machine. Fill name of disk, leave size as 100
8. Finally click "CREATE"

If everything is ok, instance will shortly create in the menu of all instances

## Connect to machine

1. Install gcloud utility
2. Auth into gcloud with 
```shell
gcloud auth login
```
Use your account which used in https://console.cloud.google.com/compute/instances

3. Set project with 
```shell
gcloud config set project <project_id>
```
4. Connect to instance with 
```shell
gcloud compute ssh <instance_name>
```

## Prepare machine
1. On connect you will be prompted to install nvidia drivers. Enter y
2. Install python3.10 with
```shell
sudo apt install lzma liblzma-dev libbz2-dev build-essential zlib1g-dev libncurses5-dev libgdbm-dev libnss3-dev libssl-dev libreadline-dev libffi-dev libsqlite3-dev wget libbz2-dev
wget https://www.python.org/ftp/python/3.10.0/Python-3.10.0.tgz
tar -xf Python-3.10.*.tgz
cd Python-3.10.*/
./configure --enable-optimizations
sudo make altinstall
```
3. Mount disk
```shell
sudo mkdir -p /mnt/disks/train/
sudo mount -o discard,defaults /dev/sdb /mnt/disks/train/
```
4. Disable conda env
```shell
conda deactivate
```
5. Prepare files
```shell
cd /mnt/disks/train
sudo chown -R <username> .
```

## Transfer files to disk
1. Transfer data archive to machine
```shell
gcloud compute scp ./Data.zip <instance_name>:/home/<username>
gcloud compute ssh <instance_name>
cd /mnt/disks/train
sudo cp /home/<username> .
sudo unzip ./Data.zip
```
2. Set python project. Copy all python files and `requirements.txt`. For each file do following.
    1. copy content of file
    2. nano <filename>
    3. CTRL + V
    4. CTRL + X
    5. Type y
 
Then execute following
```shell
python3.10 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Run model
1. Run `screen` and type any key. You entered parallel tab, that can be detached so you can run script
2. Run
```shell
source venv/bin/activate
```
3. Run
```shell
python gan_main.py
```
4. After script started to work you can detach from this screen with CTRL + A + D
5. You can reattach to this screen with 
```shell
screen -r
```
