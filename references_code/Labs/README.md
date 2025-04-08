# 2023_UAV-CV
1. 8 Labs, 2 tracks in total.
2. Each directory contains a README file; please refer to it for further details.
- by NYCU, Yang Sen Lin

### Challenges and Solutions:
1. **Efficiency**:
  - Challenge: The computational load of the image processing algorithm caused delays in the drone's operation.
  - Solution: After researching ray tracing techniques online and deciding against them, we ultimately designed a new algorithm with the team that reduced the computation time to 1/n of the original.

2. **Stability**:
  - Challenge: The drone's flight speed and turning were unstable.
  - Solution: Through multiple flight tests, we developed an algorithm that corrects deviations in real-time to within 3 centimeters.

## LAB01
1. OpenCV introduction
2. Python 3 & opencv installation
### work

  1.  圖片灰階與顏色濾鏡，對比與亮度![截圖 2024-01-14 凌晨12 56 31](https://github.com/tommy60718/2023_UAV-CV/assets/128281234/a3e3c00b-0f49-4ff4-a813-c05784e7fac4)

  2.  Nearest Neighbor Interpolation
  3.  Bilinear Interpolation![截圖 2024-01-14 凌晨12 57 32](https://github.com/tommy60718/2023_UAV-CV/assets/128281234/14321794-cf04-422b-a19d-f931ae8c0da5)


## LAB02
1. 邊緣偵測(filtering & Sobel Operator)
![截圖 2024-01-14 凌晨12 54 10](https://github.com/tommy60718/2023_UAV-CV/assets/128281234/b0636537-dd94-44ba-af3b-3b1475aaf908)
2. Histogram Equalization
3. Otsu threshold
![截圖 2024-01-14 凌晨12 54 04](https://github.com/tommy60718/2023_UAV-CV/assets/128281234/abee862d-913b-427b-95d1-4a1d51ca0506)


## LAB03
1. GrabCut Algorithm 
2. Background subtraction Threshold Connected component
![截圖 2024-01-14 凌晨12 51 48](https://github.com/tommy60718/2023_UAV-CV/assets/128281234/9c2b4983-5f53-45ea-925c-72f0d789eed4)
![截圖 2024-01-14 凌晨12 51 54](https://github.com/tommy60718/2023_UAV-CV/assets/128281234/c63f2ce0-19ad-4eb5-9411-30fa75bdebda)


## LAB04
1. Camera Calibration
![截圖 2024-01-14 凌晨1 01 20](https://github.com/tommy60718/2023_UAV-CV/assets/128281234/b9661507-d8df-4523-b7d7-09aa4168ad63)
3. Warping practice
![截圖 2024-01-14 凌晨1 06 19](https://github.com/tommy60718/2023_UAV-CV/assets/128281234/b5703b6b-d38d-4148-ad42-53882a834c19)

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

## LAB08
1.Using HOG(Histogram of Oriented Gradient) to detect & enclose body & face

  • image = cv2.rectangle(image, start_point, end_point, color, thickness)
  
2.Detect the size of the body

3.Detect the distance of the face
![IMG_1949](https://github.com/tommy60718/2023_UAV-CV/assets/128281234/2dc1bf31-ffa4-4b88-b2f1-549f79898574)


## LAB10 
1. 追線
    1. grey scale (LAB01)
    2. Otsu threshold (LAB02)
    3. technique to chasing the black line.
        https://www.youtube.com/watch?v=LmEcyQnfpDA
        <img width="911" alt="截圖 2024-03-10 下午1 34 02" src="https://github.com/tommy60718/2023_UAV-CV/assets/128281234/cabdf0b6-aaa2-4a96-b32f-291cc6eb5248">


