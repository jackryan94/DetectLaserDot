apt update && apt full-upgrade -y && reboot
# Install Python and Pip
apt install python3 python3-pip
# Create a Virtual-Environment OpenCV to install pip3
python3 -m venv opencv-python
source opencv-python/bin/activate
pip3 install numpy
pip3 install opencv-python
