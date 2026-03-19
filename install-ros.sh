#!/bin/bash
# ROS- 2 Humble Headless Installation for Ubuntu 22.04
set -e

echo "Setting up locale..."
sudo apt update && sudo apt install -y locales
sudo locale-gen en_US en_US.UTF-8
sudo update-locale LC_ALL=en_US.UTF-8 LANG=en_US.UTF-8
export LANG=en_US.UTF-8

echo "Ensuring Universe repository is enabled..."
sudo apt install -y software-properties-common
sudo add-apt-repository -y universe

echo "Adding ROS 2 GPG key..."
sudo apt update && sudo apt install -y curl
sudo curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key \
  -o /usr/share/keyrings/ros-archive-keyring.gpg

echo "Adding ROS 2 repository..."
echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(. /etc/os-release && echo $UBUNTU_CODENAME) main" \
  | sudo tee /etc/apt/sources.list.d/ros2.list > /dev/null

echo "Updating package lists..."
sudo apt update
sudo apt upgrade -y

echo "Installing ROS 2 Humble Desktop (full)..."
sudo apt install -y ros-humble-desktop

echo "Installing ROS 2 development tools..."
sudo apt install -y ros-dev-tools

echo "Initializing rosdep..."
sudo rosdep init || true   # may already be initialized
rosdep update

echo "Adding ROS 2 environment to .bashrc..."
grep -qxF 'source /opt/ros/humble/setup.bash' ~/.bashrc \
  || echo 'source /opt/ros/humble/setup.bash' >> ~/.bashrc

echo ""
echo "============================================"
echo "ROS 2 Humble installation complete!"
echo "============================================"
echo "To use ROS 2 in a new terminal, run:"
echo "  source /opt/ros/humble/setup.bash"
echo "Or open a new terminal (it will auto-source)."
echo ""
echo "Test with:"
echo "  ros2 run demo_nodes_cpp talker"
echo "  ros2 run demo_nodes_py listener"
