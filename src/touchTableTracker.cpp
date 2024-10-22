#include "touchTableTracker.h"

void TouchTableTracker::setup(const cv::Rect &track)
{
	startedNasent = ofGetElapsedTimef();
	color.setHsb(ofRandom(0, 255), 255, 255);
	cur = ofxCv::toOf(track).getCenter();
	smooth = cur;
	state = NASENT;
	trail.clear();
}

void TouchTableTracker::update(const cv::Rect &track)
{
	if (state == BORN)
	{
		state = ALIVE;
	}

	if (state == ALIVE)
	{
		startedDying = 0;

		cur = ofxCv::toOf(track).getCenter();
		smooth.interpolate(cur, .5);
		trail.addVertex(smooth);
		if (trail.size() > 50)
		{
			trail.removeVertex(0);
		}
	}
	else if (state == NASENT)
	{
		float curTime = ofGetElapsedTimef();
		if (curTime - startedNasent > nasentTime)
		{
			color.setHsb(ofRandom(0, 255), 255, 255);
			state = BORN;
		}
	}
}

void TouchTableTracker::kill()
{
	float curTime = ofGetElapsedTimef();
	if (state == ALIVE)
	{
		if (startedDying == 0)
		{
			startedDying = curTime;
		}
		else if (curTime - startedDying > dyingTime)
		{
			state = DEAD;
		}
	}
	else
	{
		state = DEAD;
	}
}

void TouchTableTracker ::draw()
{
	if (state != ALIVE)
		return;

	float curTime = ofGetElapsedTimef();
	float age = curTime - nasentTime;

	ofPushStyle();
	float size = 16;
	ofSetColor(0, 0, 255);
	if (startedDying)
	{
		ofSetColor(ofColor::red);
		size = ofMap(ofGetElapsedTimef() - startedDying, 0, dyingTime, size, 0, true);
	}
	ofNoFill();
	unsigned int label = this->getLabel();
	ofSeedRandom(label << 24);
	ofSetColor(0, 0, 255);

	ofDrawCircle(cur, size);
	ofDrawBitmapString(ofToString(label), cur);

	ofSetColor(0, 255, 255);
	switch (state)
	{
	case TouchTableTracker::NASENT:
		ofDrawBitmapString("NASENT", cur.x, cur.y - 10);
		break;
	case TouchTableTracker::BORN:
		ofDrawBitmapString("BORN", cur.x, cur.y - 10);
		break;
	case TouchTableTracker::ALIVE:
		ofDrawBitmapString("ALIVE", cur.x, cur.y - 10);
		break;
	case TouchTableTracker::DEAD:
		ofDrawBitmapString("DEAD", cur.x, cur.y - 10);
		break;
	default:
		break;
	}
	trail.draw();

	ofPopStyle();
}

void TouchTableTracker::terminate()
{
	dead = true;
	state = DEAD;
	trail.clear();
}

//--public method--------------------------------------------------------------
void TouchTableThread::getCameraImage(ofImage &image)
{
	lock();
	if (!resultImg.empty())
	{
		ofxCv::toOf(resultImg, image);
	}
	unlock();
}
void TouchTableThread ::adjustGamma(cv::Mat &img, float gamma)
{
	/*cv::Mat lookUpTable(1, 256, CV_8U);
	unsigned char* p = lookUpTable.ptr();
	for (int i = 0; i < 256; i++) {
		p[i] = cv::saturate_cast<unsigned char>(pow(i / 255.0, gamma) * 255.0);
	}*/
	uchar LUT[256];
	for (int i = 0; i < 256; i++)
	{
		LUT[i] = (int)(pow((double)i / 255.0, gamma) * 255.0);
	}
	cv::Mat lookUpTable = cv::Mat(1, 256, CV_8UC1, LUT);
	cv::LUT(img, lookUpTable, img);
}

void TouchTableThread::getWindowSize(int w_, int h_)
{
	w = w_;
	h = h_;
}

void TouchTableThread::setCamera(ofVideoGrabber *cam)
{
	camera = cam;
}

void TouchTableThread::draw()
{
	lock();
	ofSetColor(255);
	ofSetLineWidth(2);
	contourFinder_->draw();

	vector<TouchTableTracker> &followers = tracker_->getFollowers();
	for (int i = 0; i < followers.size(); i++)
	{
		followers[i].draw();
	}

	drawSrcCircle();
	unlock();
}

void TouchTableThread::setParam(
		float minAR,
		float maxAR,
		float th,
		float gm)
{
	minAreaRadius = minAR;
	maxAreaRadius = maxAR;
	threshold = th;
	gamma = gm;
	lock();
	contourFinder_->setThreshold(threshold);
	contourFinder_->setMinAreaRadius(minAreaRadius);
	contourFinder_->setMaxAreaRadius(maxAreaRadius);
	tracker_->setPersistence(30);
	tracker_->setMaximumDistance(60);
	unlock();
}

//--private method-------------------------------------------------
void TouchTableThread::threadedFunction()
{
	contourFinder_->setAutoThreshold(false); // 自動閾値設定をオフにする
	contourFinder_->setInvert(false);				 // 反転をオフにする
	while (isThreadRunning())
	{
		lock();
		if (camera != nullptr && camera->isFrameNew())
		{
			img = ofxCv::toCv(*camera);
			if (!img.empty())
			{
				cv::Mat hsv;
				cv::cvtColor(img, hsv, cv::COLOR_RGB2HSV);

				// 黒色の範囲を定義
				cv::Scalar lowerBlack = cv::Scalar(0, 0, 0);
				cv::Scalar upperBlack = cv::Scalar(180, 255, 50); // 明度を調整して黒の範囲を設定

				cv::Mat blackMask;
				cv::inRange(hsv, lowerBlack, upperBlack, blackMask);

				if (!perspectiveMat.empty() && perspectiveMat.size() == cv::Size(3, 3))
				{
					cv::warpPerspective(blackMask, blackMask, perspectiveMat, blackMask.size(), cv::INTER_NEAREST);
				}

				cv::GaussianBlur(blackMask, blackMask, cv::Size(11, 11), 0, 0);
				adjustGamma(blackMask, gamma);

				resultImg = (isCalibMode) ? img : blackMask.clone();
				contourFinder_->setThreshold(threshold); // 閾値を設定
				contourFinder_->findContours(blackMask);
				tracker_->track(contourFinder_->getBoundingRects());
				sendContourData(); // 毎フレーム送信
			}
		}
		unlock();
		ofSleepMillis(2);
	}
}

//--Circle Drawing for Perspective

void TouchTableThread::reset_Circle()
{
	pts_src.clear();
	pts_src.push_back(ofVec2f(0, 0));
	pts_src.push_back(ofVec2f(w - 1, 0));
	pts_src.push_back(ofVec2f(w - 1, h - 1));
	pts_src.push_back(ofVec2f(0, h - 1));
	setPerspective(pts_src);
}

void TouchTableThread::setCalibMode(bool calibMode)
{
	if (isCalibMode != calibMode)
	{
		lock();
		isCalibMode = calibMode;
		pickedCircle = -1;
		unlock();
	}
}

bool TouchTableThread::getCalibMode()
{
	return isCalibMode;
}

void TouchTableThread::drawSrcCircle()
{
	if (isCalibMode)
	{
		ofPushStyle();
		ofNoFill();
		ofSetColor(255, 0, 0);
		ofDrawCircle(pts_src[0], 10);
		ofDrawBitmapString("0", pts_src[0]);
		ofDrawLine(pts_src[0], pts_src[1]);

		ofSetColor(0, 255, 0);
		ofDrawCircle(pts_src[1], 10);
		ofDrawBitmapString("1", pts_src[1]);
		ofDrawLine(pts_src[1], pts_src[2]);

		ofSetColor(0, 0, 255);
		ofDrawCircle(pts_src[2], 10);
		ofDrawBitmapString("2", pts_src[2]);
		ofDrawLine(pts_src[2], pts_src[3]);

		ofSetColor(255, 255, 0);
		ofDrawCircle(pts_src[3], 10);
		ofDrawBitmapString("3", pts_src[3]);
		ofDrawLine(pts_src[3], pts_src[0]);
		ofFill();
		ofPopStyle();
	}
}

void TouchTableThread::moveClosestPoint(int x, int y)
{
	if (pickedCircle > -1)
	{
		pts_src[pickedCircle] = ofVec2f(x, y) + pickOffset;
	}
}

void TouchTableThread::pickClosestPoint(int x, int y)
{
	float cls = INFINITY;
	for (size_t i = 0; i < 4; i++)
	{
		ofVec2f v = pts_src[i] - ofVec2f(x, y);
		float d = v.length();
		if (cls > d)
		{
			pickedCircle = i;
			cls = d;
			pickOffset = v;
		}
	}
}

void TouchTableThread::setPerspective(std::vector<ofVec2f> circles)
{
	if (circles.size() != 4)
		return;

	pts_src.clear();
	std::vector<cv::Point2f> src;
	for (size_t i = 0; i < 4; i++)
	{
		pts_src.push_back(circles[i]);
		src.push_back(cv::Point2f(circles[i].x, circles[i].y));
	}

	std::vector<cv::Point2f> dst;
	dst.push_back(cv::Point2f(0, 0));
	dst.push_back(cv::Point2f(w - 1, 0));
	dst.push_back(cv::Point2f(w - 1, h - 1));
	dst.push_back(cv::Point2f(0, h - 1));

	lock();
	if (src.size() == 4 && dst.size() == 4)
	{
		perspectiveMat = cv::getPerspectiveTransform(src, dst);
	}
	else
	{
		ofLogError("TouchTableThread") << "Invalid number of points for perspective transform";
	}
	unlock();
}

void TouchTableThread::setCalib()
{
	setPerspective(pts_src);
}

void TouchTableThread::setupSocket(const std::string &address, int port)
{
#ifdef _WIN32
	WSADATA wsaData;
	if (WSAStartup(MAKEWORD(2, 2), &wsaData) != 0)
	{
		ofLogError("TouchTableThread") << "WSAStartup failed";
		socketConnected = false;
		return;
	}

	socketFd = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
	if (socketFd == INVALID_SOCKET)
#else
	socketFd = socket(AF_INET, SOCK_STREAM, 0);
	if (socketFd < 0)
#endif
	{
		ofLogError("TouchTableThread") << "socket create is failed";
		socketConnected = false;
		return;
	}

	struct sockaddr_in serverAddr;
	serverAddr.sin_family = AF_INET;
	serverAddr.sin_port = htons(port);
	if (inet_pton(AF_INET, address.c_str(), &serverAddr.sin_addr) <= 0)
	{
		ofLogError("TouchTableThread") << "address is not supported";
#ifdef _WIN32
		closesocket(socketFd);
		WSACleanup();
#else
		close(socketFd);
#endif
		socketConnected = false;
		return;
	}

	if (connect(socketFd, (struct sockaddr *)&serverAddr, sizeof(serverAddr)) < 0)
	{
		ofLogError("TouchTableThread") << "server connection is failed";
#ifdef _WIN32
		closesocket(socketFd);
		WSACleanup();
#else
		close(socketFd);
#endif
		socketConnected = false;
		return;
	}

	socketConnected = true;
	ofLogNotice("TouchTableThread") << "socket connect succeed: " << address << ":" << port;
}

void TouchTableThread::sendContourData()
{
	if (!socketConnected)
		return;

	nlohmann::json root;
	nlohmann::json contours = nlohmann::json::array();

	for (int i = 0; i < contourFinder_->size(); i++)
	{
		nlohmann::json contour;
		std::vector<cv::Point> points = contourFinder_->getContour(i);
		nlohmann::json vertices = nlohmann::json::array();

		for (const auto &point : points)
		{
			nlohmann::json vertex;
			vertex["x"] = w / 2.0f - point.x;
			vertex["y"] = h / 2.0f - point.y;
			vertices.push_back(vertex);
		}

		contour["vertices"] = vertices;
		contours.push_back(contour);
	}

	root["contours"] = contours;

	std::string json_str = root.dump() + "\n"; // 改行を追加

#ifdef _WIN32
	int bytesSent = send(socketFd, json_str.c_str(), json_str.length(), 0);
	if (bytesSent == SOCKET_ERROR)
#else
	ssize_t bytesSent = send(socketFd, json_str.c_str(), json_str.length(), 0);
	if (bytesSent < 0)
#endif
	{
		ofLogError("TouchTableThread") << "データ送信エラー";
		socketConnected = false;
#ifdef _WIN32
		closesocket(socketFd);
#else
		close(socketFd);
#endif
	}
}
