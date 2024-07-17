#include "Blob.h"
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>
#include <vector>
#include <cmath>

#define SHOW_STEPS // un-comment | comment this line to show steps or not

// const global variables
const cv::Scalar SCALAR_BLACK = cv::Scalar(0.0, 0.0, 0.0);
const cv::Scalar SCALAR_WHITE = cv::Scalar(255.0, 255.0, 255.0);
const cv::Scalar SCALAR_YELLOW = cv::Scalar(0.0, 255.0, 255.0);
const cv::Scalar SCALAR_GREEN = cv::Scalar(0.0, 200.0, 0.0);
const cv::Scalar SCALAR_RED = cv::Scalar(0.0, 0.0, 255.0);

// function prototypes
void matchCurrentFrameBlobsToExistingBlobs(std::vector<Blob> &existingBlobs, std::vector<Blob> &currentFrameBlobs);
void addBlobToExistingBlobs(Blob &currentFrameBlob, std::vector<Blob> &existingBlobs, int &intIndex);
void addNewBlob(Blob &currentFrameBlob, std::vector<Blob> &existingBlobs);
double distanceBetweenPoints(cv::Point point1, cv::Point point2);
void drawAndShowContours(cv::Size imageSize, std::vector<std::vector<cv::Point>> contours, std::string strImageName);
void drawAndShowContours(cv::Size imageSize, std::vector<Blob> blobs, std::string strImageName);
bool checkIfBlobsCrossedTheLineLeft(std::vector<Blob> &blobs, int &intHorizontalLinePosition, int &carCountLeft);
void drawBlobInfoOnImage(std::vector<Blob> &blobs, cv::Mat &imgFrame2Copy);
void drawCarCountOnImage(int &carCountLeft, cv::Mat &imgFrame2Copy);

// global variables
int carCountLeft = 0;

int main(void) {
    cv::VideoCapture capVideo;
    cv::Mat imgFrame1, imgFrame2;
    std::vector<Blob> blobs;
    cv::Point crossingLineLeft[2];

    capVideo.open("../../src/highway_online.mp4");

    if (!capVideo.isOpened()) {
        std::cout << "Error reading video file" << std::endl;
        return 0;
    }

    if (capVideo.get(cv::CAP_PROP_FRAME_COUNT) < 2) {
        std::cout << "Error: video file must have at least two frames";
        return 0;
    }

    capVideo.read(imgFrame1);
    capVideo.read(imgFrame2);

    // Resize frames to a lower resolution (e.g., 1280x720)
    cv::resize(imgFrame1, imgFrame1, cv::Size(1000, 750));
    cv::resize(imgFrame2, imgFrame2, cv::Size(1000, 750));

    // Control line for car count (left way)
    int intHorizontalLinePosition = static_cast<int>(std::round(imgFrame1.rows * 0.35 * 1.40));

    crossingLineLeft[0] = cv::Point(0, intHorizontalLinePosition);
    crossingLineLeft[1] = cv::Point(imgFrame1.cols - 1, intHorizontalLinePosition);

    char chCheckForEscKey = 0;
    bool blnFirstFrame = true;
    int frameCount = 2;

    while (capVideo.isOpened() && chCheckForEscKey != 27) {
        std::vector<Blob> currentFrameBlobs;
        cv::Mat imgFrame1Copy = imgFrame1.clone();
        cv::Mat imgFrame2Copy = imgFrame2.clone();
        cv::Mat imgDifference, imgThresh;

        cv::cvtColor(imgFrame1Copy, imgFrame1Copy, cv::COLOR_BGR2GRAY);
        cv::cvtColor(imgFrame2Copy, imgFrame2Copy, cv::COLOR_BGR2GRAY);

        cv::GaussianBlur(imgFrame1Copy, imgFrame1Copy, cv::Size(5, 5), 0);
        cv::GaussianBlur(imgFrame2Copy, imgFrame2Copy, cv::Size(5, 5), 0);

        cv::absdiff(imgFrame1Copy, imgFrame2Copy, imgDifference);
        cv::threshold(imgDifference, imgThresh, 30, 255.0, cv::THRESH_BINARY);

        cv::imshow("imgThresh", imgThresh);
        cv::resizeWindow("imgThresh", 1000, 750); // Resize the window

        cv::Mat structuringElement5x5 = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(5, 5));

        for (int i = 0; i < 2; i++) {
            cv::dilate(imgThresh, imgThresh, structuringElement5x5);
            cv::erode(imgThresh, imgThresh, structuringElement5x5);
        }

        cv::Mat imgThreshCopy = imgThresh.clone();
        std::vector<std::vector<cv::Point>> contours;

        cv::findContours(imgThreshCopy, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
        drawAndShowContours(imgThresh.size(), contours, "imgContours");
        cv::resizeWindow("imgContours", 1000, 750); // Resize the window

        std::vector<std::vector<cv::Point>> convexHulls(contours.size());

        for (size_t i = 0; i < contours.size(); i++) {
            cv::convexHull(contours[i], convexHulls[i]);
        }

        drawAndShowContours(imgThresh.size(), convexHulls, "imgConvexHulls");
        cv::resizeWindow("imgConvexHulls", 1000, 750); // Resize the window

        for (auto &convexHull : convexHulls) {
            Blob possibleBlob(convexHull);

            if (possibleBlob.currentBoundingRect.area() > 400 &&
                possibleBlob.dblCurrentAspectRatio > 0.2 &&
                possibleBlob.dblCurrentAspectRatio < 4.0 &&
                possibleBlob.currentBoundingRect.width > 30 &&
                possibleBlob.currentBoundingRect.height > 30 &&
                possibleBlob.dblCurrentDiagonalSize > 60.0 &&
                (cv::contourArea(possibleBlob.currentContour) / possibleBlob.currentBoundingRect.area()) > 0.50) {
                currentFrameBlobs.push_back(possibleBlob);
            }
        }

        drawAndShowContours(imgThresh.size(), currentFrameBlobs, "imgCurrentFrameBlobs");
        cv::resizeWindow("imgCurrentFrameBlobs", 1000, 750); // Resize the window

        if (blnFirstFrame) {
            blobs = currentFrameBlobs;
            blnFirstFrame = false;
        } else {
            matchCurrentFrameBlobsToExistingBlobs(blobs, currentFrameBlobs);
        }

        drawAndShowContours(imgThresh.size(), blobs, "imgBlobs");
        cv::resizeWindow("imgBlobs", 1000, 750); // Resize the window

        imgFrame2Copy = imgFrame2.clone();
        drawBlobInfoOnImage(blobs, imgFrame2Copy);

        // Check the left way
        bool blnAtLeastOneBlobCrossedTheLineLeft = checkIfBlobsCrossedTheLineLeft(blobs, intHorizontalLinePosition, carCountLeft);

        // Left way
        cv::line(imgFrame2Copy, crossingLineLeft[0], crossingLineLeft[1], blnAtLeastOneBlobCrossedTheLineLeft ? SCALAR_WHITE : SCALAR_YELLOW, 2);

        drawCarCountOnImage(carCountLeft, imgFrame2Copy);
        cv::imshow("imgFrame2Copy", imgFrame2Copy);
        cv::resizeWindow("imgFrame2Copy", 1000, 750); // Resize the window

        // Prepare for the next iteration
        currentFrameBlobs.clear();
        imgFrame1 = imgFrame2.clone();

               if ((capVideo.get(cv::CAP_PROP_POS_FRAMES) + 1) < capVideo.get(cv::CAP_PROP_FRAME_COUNT)) {
            capVideo.read(imgFrame2);

            // Resize the frame to a lower resolution (e.g., 1280x720)
            cv::resize(imgFrame2, imgFrame2, cv::Size(1000, 750));
        } else {
            std::cout << "End of video\n";
            break;
        }

        chCheckForEscKey = cv::waitKey(70);
        frameCount++;
    }

    if (chCheckForEscKey != 27) { // if the user did not press 'esc' key
        cv::waitKey(0); // hold the windows open until user presses a key
    }

    return 0;
}

void matchCurrentFrameBlobsToExistingBlobs(std::vector<Blob> &existingBlobs, std::vector<Blob> &currentFrameBlobs) {
    for (auto &existingBlob : existingBlobs) {
        existingBlob.blnCurrentMatchFoundOrNewBlob = false;
        existingBlob.predictNextPosition();
    }

    for (auto &currentFrameBlob : currentFrameBlobs) {
        int intIndexOfLeastDistance = 0;
        double dblLeastDistance = 100000.0;

        for (unsigned int i = 0; i < existingBlobs.size(); i++) {
            if (existingBlobs[i].blnStillBeingTracked) {
                double dblDistance = distanceBetweenPoints(currentFrameBlob.centerPositions.back(), existingBlobs[i].predictedNextPosition);

                if (dblDistance < dblLeastDistance) {
                    dblLeastDistance = dblDistance;
                    intIndexOfLeastDistance = i;
                }
            }
        }

        if (dblLeastDistance < currentFrameBlob.dblCurrentDiagonalSize * 0.5) {
            addBlobToExistingBlobs(currentFrameBlob, existingBlobs, intIndexOfLeastDistance);
        } else {
            addNewBlob(currentFrameBlob, existingBlobs);
        }
    }

    for (auto &existingBlob : existingBlobs) {
        if (!existingBlob.blnCurrentMatchFoundOrNewBlob) {
            existingBlob.intNumOfConsecutiveFramesWithoutAMatch++;
        }

        if (existingBlob.intNumOfConsecutiveFramesWithoutAMatch >= 5) {
            existingBlob.blnStillBeingTracked = false;
        }
    }
}

void addBlobToExistingBlobs(Blob &currentFrameBlob, std::vector<Blob> &existingBlobs, int &intIndex) {
    existingBlobs[intIndex].currentContour = currentFrameBlob.currentContour;
    existingBlobs[intIndex].currentBoundingRect = currentFrameBlob.currentBoundingRect;
    existingBlobs[intIndex].centerPositions.push_back(currentFrameBlob.centerPositions.back());
    existingBlobs[intIndex].dblCurrentDiagonalSize = currentFrameBlob.dblCurrentDiagonalSize;
    existingBlobs[intIndex].dblCurrentAspectRatio = currentFrameBlob.dblCurrentAspectRatio;
    existingBlobs[intIndex].blnStillBeingTracked = true;
    existingBlobs[intIndex].blnCurrentMatchFoundOrNewBlob = true;
}

void addNewBlob(Blob &currentFrameBlob, std::vector<Blob> &existingBlobs) {
    currentFrameBlob.blnCurrentMatchFoundOrNewBlob = true;
    existingBlobs.push_back(currentFrameBlob);
}

double distanceBetweenPoints(cv::Point point1, cv::Point point2) {
    int intX = abs(point1.x - point2.x);
    int intY = abs(point1.y - point2.y);

    return sqrt(pow(intX, 2) + pow(intY, 2));
}

void drawAndShowContours(cv::Size imageSize, std::vector<std::vector<cv::Point>> contours, std::string strImageName) {
    cv::Mat image(imageSize, CV_8UC3, SCALAR_BLACK);
    cv::drawContours(image, contours, -1, SCALAR_WHITE, -1);
    cv::imshow(strImageName, image);
    cv::resizeWindow(strImageName, 1000, 750); // Resize the window
}

void drawAndShowContours(cv::Size imageSize, std::vector<Blob> blobs, std::string strImageName) {
    cv::Mat image(imageSize, CV_8UC3, SCALAR_BLACK);
    std::vector<std::vector<cv::Point>> contours;

    for (auto &blob : blobs) {
        if (blob.blnStillBeingTracked) {
            contours.push_back(blob.currentContour);
        }
    }

    cv::drawContours(image, contours, -1, SCALAR_WHITE, -1);
    cv::imshow(strImageName, image);
    cv::resizeWindow(strImageName, 1000, 750); // Resize the window
}

bool checkIfBlobsCrossedTheLineLeft(std::vector<Blob> &blobs, int &intHorizontalLinePosition, int &carCountLeft) {
    bool blnAtLeastOneBlobCrossedTheLineLeft = false;

    for (auto &blob : blobs) {
        if (blob.blnStillBeingTracked && blob.centerPositions.size() >= 2) {
            int prevFrameIndex = (int)blob.centerPositions.size() - 2;
            int currFrameIndex = (int)blob.centerPositions.size() - 1;

            if (blob.centerPositions[prevFrameIndex].y <= intHorizontalLinePosition &&
                blob.centerPositions[currFrameIndex].y > intHorizontalLinePosition) {
                carCountLeft++;
                blnAtLeastOneBlobCrossedTheLineLeft = true;
            }
        }
    }

    return blnAtLeastOneBlobCrossedTheLineLeft;
}

void drawBlobInfoOnImage(std::vector<Blob> &blobs, cv::Mat &imgFrame2Copy) {
    for (unsigned int i = 0; i < blobs.size(); i++) {
        if (blobs[i].blnStillBeingTracked) {
            cv::rectangle(imgFrame2Copy, blobs[i].currentBoundingRect, SCALAR_RED, 1);

            int intFontFace = cv::FONT_HERSHEY_SIMPLEX;
            double dblFontScale = (imgFrame2Copy.rows * imgFrame2Copy.cols) / 150000.0;
            int intFontThickness = (int)std::round(dblFontScale * 1.0);

            cv::putText(imgFrame2Copy, std::to_string(i), blobs[i].centerPositions.back(), intFontFace, dblFontScale, SCALAR_GREEN, intFontThickness);
        }
    }
}

void drawCarCountOnImage(int &carCountLeft, cv::Mat &imgFrame2Copy) {
    int intFontFace = cv::FONT_HERSHEY_SIMPLEX;
    double dblFontScale = (imgFrame2Copy.rows * imgFrame2Copy.cols) / 300000.0;
    int intFontThickness = (int)std::round(dblFontScale * 1.0);

    // Left way
    cv::putText(imgFrame2Copy, "Vehicle count: " + std::to_string(carCountLeft), cv::Point(10, 25), intFontFace, dblFontScale, SCALAR_YELLOW, intFontThickness);
}
