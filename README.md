# Vehicle Counting with OpenCV and C++

This project is a C++ application for counting vehicles crossing a predefined line in a video stream using OpenCV-based motion detection, contour extraction, and blob tracking.

## Overview

The program:

- reads a road traffic video,
- compares consecutive frames,
- detects moving objects by difference thresholding,
- filters contours that look like vehicles,
- tracks blobs across frames,
- counts vehicles crossing a detection line.

The main implementation is in [src/Main.cpp](src/Main.cpp), with supporting blob logic in [src/Blob.h](src/Blob.h) and [src/Blob.cpp](src/Blob.cpp).

## Project Structure

```text
.
├── CMakeLists.txt
├── README.md
├── build/                     # generated build output
└── src/
    ├── Blob.cpp
    ├── Blob.h
    ├── CMakeLists.txt
    ├── Main.cpp
    └── highway_online.mp4     # expected video input
```

## Requirements

- CMake
- C++11 compatible compiler
- OpenCV development libraries

A typical Linux installation is:

```bash
sudo apt update
sudo apt install build-essential cmake libopencv-dev
```

## Building the Project

From the project root:

```bash
mkdir -p build
cd build
cmake ..
cmake --build .
```

This creates the executable:

```text
build/src/vehicle_counting_demo
```

## Running the Demo

The application expects the video file to be available at:

```text
src/highway_online.mp4
```

Run it from the build output directory so the relative video path resolves correctly:

```bash
cd build/src
./vehicle_counting_demo
```

Press `Esc` to exit the application.

## Notes

- The program uses a fixed relative path to the input video: `../../src/highway_online.mp4`.
- If the video file is missing, the app will fail to open the stream.
- The app displays several intermediate OpenCV windows such as thresholded frames, contours, and tracked blobs.

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for the full text.
