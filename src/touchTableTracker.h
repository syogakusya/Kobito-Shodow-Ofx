#pragma once
#include "ofMain.h"
#include "ofxCv.h"
#include <thread>
#include <mutex>
#include <atomic>
#include <string>
#include <vector>
#include <json.hpp> // nlohmann/json ライブラリを使用

#include <opencv2/imgproc.hpp>

#ifdef _WIN32
#include <winsock2.h>
#include <ws2tcpip.h>
#pragma comment(lib, "ws2_32.lib")
#else
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <unistd.h>
#endif

class TouchTableTracker : public ofxCv::RectFollower
{
protected:
	ofColor color;
	float startedDying, startedNasent, dyingTime, nasentTime;
	ofPolyline trail;

public:
	ofVec3f cur, smooth;
	enum
	{
		NASENT,
		BORN,
		ALIVE,
		DEAD
	} state;

	TouchTableTracker() : startedDying(0),
												startedNasent(0),
												dyingTime(0.5),
												nasentTime(0.2) {}

	void setup(const cv::Rect &track);
	void update(const cv::Rect &track);
	void kill();
	void draw();
	void terminate();
};

class TouchTableThread : public ofThread
{
public:
	TouchTableThread()
	{
		perspectiveMat = cv::Mat::eye(3, 3, CV_64F);
		contourFinder_ = std::make_unique<ofxCv::ContourFinder>();
		tracker_ = std::make_unique<ofxCv::RectTrackerFollower<TouchTableTracker>>();
	};

	~TouchTableThread()
	{
		waitForThread(true);
	}

	void getWindowSize(int w_, int h_);
	void setCamera(ofVideoGrabber *cam);
	void adjustGamma(cv::Mat &img, float gamma = 1.0);
	void draw();
	void setParam(
			float minAR,
			float maxAR,
			float th,
			float gm,
			int br);
	void getCameraImage(ofImage &image);

	void setupSocket(const std::string &address, int port);
	void sendContourData();

private:
	ofVideoGrabber *camera = nullptr;
	std::unique_ptr<ofxCv::ContourFinder> contourFinder_;
	std::unique_ptr<ofxCv::RectTrackerFollower<TouchTableTracker>> tracker_;
	cv::Mat img, gray;

	// contourFinder param
	float minAreaRadius;
	float maxAreaRadius;
	float gamma;
	float threshold;

	int w, h;

	void threadedFunction();

	//--Circle Drawing for Perspective--------------------------------------------
public:
	std::vector<ofVec2f> pts_src;

	void reset_Circle();
	void setCalibMode(bool calibMode);
	bool getCalibMode();
	void moveClosestPoint(int x, int y);
	void pickClosestPoint(int x, int y);
	void setCalib();
	void setPerspective(std::vector<ofVec2f> circles);

private:
	bool isCalibMode;
	int pickedCircle;
	int maxBrightness;
	ofVec2f pickOffset;
	cv::Mat perspectiveMat;
	cv::Mat resultImg;
	void drawSrcCircle();

#ifdef _WIN32
	SOCKET socketFd;
#else
	int socketFd;
#endif
	std::atomic<bool> socketConnected;
};
