## LAB05
1. Aruco marker dection
   1. calibrate the drone camera
   2. marker detection by drone camera
   3. pose estimation ![截圖 2024-01-14 凌晨1 13 53](https://github.com/tommy60718/2023_UAV-CV/assets/128281234/1de20708-e45a-487b-9ec9-27fc3c3b3d5f)
2. Tello drone (distance detection)
3. ![截圖 2024-01-14 凌晨1 17 25](https://github.com/tommy60718/2023_UAV-CV/assets/128281234/7fd35d1e-c6a8-4e4a-adc3-aae4cd808ec1)

## LAB06
1. 無人機手動控制
2. 無人機自動追蹤
    1. Calibration
    2. PID control
    3. Using Tello drone functions: drone.send_rc_control(x_update,z_update,y_update,yaw_update)

## LAB07 & midterm
![截圖 2024-01-14 凌晨1 22 16](https://github.com/tommy60718/2023_UAV-CV/assets/128281234/1e6ddc58-28b6-43f6-b8f1-f00111367d7f)
![截圖 2024-01-14 凌晨1 22 20](https://github.com/tommy60718/2023_UAV-CV/assets/128281234/ee5da9bf-e557-45cd-804f-eeda30594e15)
![截圖 2024-01-14 凌晨1 22 23](https://github.com/tommy60718/2023_UAV-CV/assets/128281234/bca06d9c-c30c-4118-881f-f476f39d2d0c)
![截圖 2024-01-14 凌晨1 22 26](https://github.com/tommy60718/2023_UAV-CV/assets/128281234/c5703919-a2d1-400c-97d5-31034b8912e2)

## LAB10 
1. 追線
    1. grey scale (LAB01)
    2. Otsu threshold (LAB02)
    3. technique to chasing the black line.
        https://www.youtube.com/watch?v=LmEcyQnfpDA
       ![截圖 2024-01-14 凌晨1 28 08](https://github.com/tommy60718/2023_UAV-CV/assets/128281234/214a243a-2d42-4bc3-b3d3-cde5396709dc)



# Tello-Video

This is an example using the Tello SDK v1.3.0.0 and above to receive video stream from Tello camera,decode the video stream and show the image by GUI.

 - Written in Python 2.7
 - Tello SDK v1.3.0.0 and above(with h.264 video streaming)
 - This example includes a simple UI build with Tkinter to interact with Tello
 - Interactive control of Tello based on human movement is achieved via body pose recognition module.

## Prerequisites

- Python2.7
- pip
- Python OpenCV
- Numpy 
- PIL
- libboost-python
- Tkinter
- homebrew(for mac)
- Python h264 decoder
    - <https://github.com/DaWelter/h264decoder>

## Installation

In order to facilitate you to install python2.7 and various dependencies, we have written a one-click installation script for windows, Linux and macos. You can choose to run this script for the one-click installation, or you can download python2.7 and related libraries and dependencies online. If you have questions about the actions that the script performs, you can open the script with an editor and look up the comments for each instruction in the script. In addition, we have additionally written an uninstall script that cleans and restores all downloaded and configured content from the one-click installation script.

- **Windows**

    Go to the "install\Windows" folder,select and run the correct  "windows_install.bat" according to your computer operating system bits. 

- **Linux (Ubuntu 14.04 and above)**

    Go to the "install\Linux" folder in command line, run
    
    ```
    chmod +x linux_install.sh
    ./linux_install.sh
    ```

- **Mac**

   1. Make sure you have the latest Xcode command line tools installed. If not, you might need to update your OS X and XCode to the latest version in order to compile the h264 decoder module
   2. Go to the "install\Mac" folder folder in command line, run
   
  ```
     chmod a+x ./mac_install.sh
     ./mac_install.sh
  ```
    
    If you see no errors during installation, you are good to go!

## Run the project
- **Step1**. Turn on Tello and connect your computer device to Tello via wifi.


- **Step2**. Open project folder in terminal. Run:
    
    ```
    python main.py
    ```

- **Step3**. A UI will show up, you can now:

    - Watch live video stream from the Tello camera;
    - Take snapshot and save jpg to local folder;
    - Open Command Panel, which allows you to:
        - Take Off
        - Land
        - Flip (in forward, backward, left and right direction)
        - Control Tello using keyboard inputs:
            - **[key-Up]** move forward 20cm
            - **[key-Down]** move backward 20cm
            - **[key-Left]** move left 20 cm
            - **[key-Right]** move right 20 cm
            - **[key-w]** move up 20cm
            - **[key-s]** move down 20cm
            - **[key-a]** rotate counter-clockwise by 30 degree
            - **[key-d]** rotate clockwise by 30 degree
        -  You can also adjust the **distance** and **degree** via the trackbar and hit the "reset distance" or "reset degree" button to customize your own control.
    
## Project Description

### tello.py - class Tello

Wrapper class to interact with Tello drone.
Modified from <https://github.com/microlinux/tello>

The object starts 2 threads:

 1. thread for receiving command response from Tello
 2. thread for receiving video stream

You can use **read()** to read the last frame from Tello camera, and pause the video by setting **video_freeze(is_freeze=True)**.

### tello_control_ui.py - class TelloUI

Modified from: https://www.pyimagesearch.com/2016/05/30/displaying-a-video-feed-with-opencv-and-tkinter/

Build with Tkinter. Display video, control video play/pause and control Tello using buttons and arrow keys.

### h264decoder - class libh264decoder

From <https://github.com/DaWelter/h264decoder>.

A c++ based class that decodes raw h264 data. This module interacts with python language via python-libboost library, and its decoding functionality is based on ffmpeg library. 

After compilation, a libh264decoder.so or libh264decoder.pyd file will be placed in the working directory so that the main python file can reference it. 

If you have to compile it from source,with Linux or Mac,you can:

```
cd h264decoder
mkdir build
cd build
cmake ..
make
cp libh264decoder.so ../../
```
With Windows,you can create a project through visual studio, add files in h264decoder and dependencies such as ffmpeg and libboost, compile the project and generate a libh264decoder.pyd file.We have generated a libh264decoder.pyd and put it in the "\h264decoder\Windows"foleder so that you can copy put it to "python/site-package".

##Contact Information

If you have any questions about this sample code and the installation, please feel free to contact me. You can communicate with me by sending e-mail to sdk@ryzerobotics.com.
