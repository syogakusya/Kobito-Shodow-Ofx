#include "ofApp.h"

void ofApp::getWindowSize(int w_, int h_)
{
	w = w_;
	h = h_;
}

//--------------------------------------------------------------
void ofApp::setup()
{
	ofSetLogLevel(OF_LOG_VERBOSE);
	cam.setVerbose(true);
	cam.listDevices();

	ofSetVerticalSync(true);
	ofBackground(0);

	cam.setup(w, h);

	gui.setup();
	// gui.setPosition(480, 20);
	gui.add(minAreaRadius_.set("MinAreaRadius", 10.0, 0, 30.0));
	gui.add(maxAreaRadius_.set("MaxAreaRadius", 100, 0, 500));
	gui.add(threshold_.set("Threshold", 128, 0, 255));
	gui.add(gamma_.set("Gamma", 0.5, 0.1, 60));
	gui.add(isCalibMode_.set("Calibration", false));

	// OpenCVの並列処理を無効化
	cv::setNumThreads(0);
	cv::ocl::setUseOpenCL(false);

	// カメラ画像のテクスチャを適切に割り当てる
	cameraImg.allocate(w, h, OF_IMAGE_COLOR);
	cameraImg.setUseTexture(true);

	touchTableTracker_ = std::make_unique<TouchTableThread>();
	touchTableTracker_->getWindowSize(w, h);
	touchTableTracker_->reset_Circle();
	touchTableTracker_->setupSocket("127.0.0.1", 2001);
	loadParam();
	touchTableTracker_->startThread(true);
}

//--------------------------------------------------------------
void ofApp::update()
{
	cam.update();
	if (cam.isFrameNew())
	{
		ofPixels &pixels = cam.getPixels();
		if (pixels.isAllocated())
		{
			cameraImg.setFromPixels(pixels);
			touchTableTracker_->getCameraImage(cameraImg);
			cameraImg.update();
		}
	}
	touchTableTracker_->setCalibMode(isCalibMode_);
	touchTableTracker_->setCamera(&cam);
	touchTableTracker_->setParam(minAreaRadius_, maxAreaRadius_, threshold_, gamma_);
}

//--------------------------------------------------------------
void ofApp::draw()
{
	// cam.draw(0, 0);
	if (cameraImg.isAllocated())
	{
		cameraImg.draw(0, 0);
	}

	touchTableTracker_->draw();

	// draw FPS
	ofPushStyle();
	ofSetColor(0, 0, 255);
	auto msg = "fps: " + ofToString(ofGetFrameRate(), 0);
	ofDrawBitmapString(msg, 500, 20);
	ofPopStyle();

	// draw GUI
	if (drawGui)
	{
		gui.draw();
	}
}

//--------------------------------------------------------------
void ofApp::exit()
{
	if (touchTableTracker_)
	{
		touchTableTracker_->stopThread();
		touchTableTracker_->waitForThread(true);
	}
	saveParam();
}

//--------------------------------------------------------------
void ofApp::keyPressed(int key)
{
	if (key == 'g')
	{
		drawGui = !drawGui;
	}
	if (key == 'r')
	{
		cam.close();
		cam.setup(w, h);
	}
}

//--------------------------------------------------------------
void ofApp::keyReleased(int key)
{
}

//--------------------------------------------------------------
void ofApp::mouseMoved(int x, int y)
{
}

//--------------------------------------------------------------
void ofApp::mouseDragged(int x, int y, int button)
{
	if (touchTableTracker_->getCalibMode())
	{
		touchTableTracker_->moveClosestPoint(x, y);
	}
}

//--------------------------------------------------------------
void ofApp::mousePressed(int x, int y, int button)
{
	if (touchTableTracker_->getCalibMode())
	{
		touchTableTracker_->pickClosestPoint(x, y);
	}
}

//--------------------------------------------------------------
void ofApp::mouseReleased(int x, int y, int button)
{
	if (touchTableTracker_->getCalibMode())
	{
		touchTableTracker_->setCalib();
	}
}

//--------------------------------------------------------------
void ofApp::mouseEntered(int x, int y)
{
}

//--------------------------------------------------------------
void ofApp::mouseExited(int x, int y)
{
}

//--------------------------------------------------------------
void ofApp::windowResized(int w, int h)
{
}

//--------------------------------------------------------------
void ofApp::gotMessage(ofMessage msg)
{
}

//--------------------------------------------------------------
void ofApp::dragEvent(ofDragInfo dragInfo)
{
}

void ofApp::loadParam()
{
	nlohmann::json j;
	std::ifstream ifs("data.json");
	if (!ifs)
		return;
	ifs >> j;
	ifs.close();

	std::vector<ofVec2f> circles;
	for (size_t i = 0; i < 4; i++)
	{
		circles.push_back(ofVec2f(j["rect"][i * 2], j["rect"][i * 2 + 1]));
	}
	touchTableTracker_->setPerspective(circles);

	threshold_ = j["tracker"]["threshold"];
	minAreaRadius_ = j["tracker"]["minAreaRadius"];
	maxAreaRadius_ = j["tracker"]["maxArearaduis"];
	gamma_ = j["tracker"]["gamma"];
}

void ofApp::saveParam()
{
	nlohmann::json j;
	std::array<float, 8> circles;
	for (size_t i = 0; i < 4; i++)
	{
		ofVec2f p = touchTableTracker_->pts_src[i];
		circles[i * 2] = p.x;
		circles[i * 2 + 1] = p.y;
	}
	j["rect"] = circles;
	j["tracker"]["threshold"] = (float)this->threshold_;
	j["tracker"]["minAreaRadius"] = (float)this->minAreaRadius_;
	j["tracker"]["maxArearaduis"] = (float)this->maxAreaRadius_;
	j["tracker"]["gamma"] = (float)this->gamma_;

	std::ofstream ofs("data.json");
	ofs << j.dump(4) << std::endl;
	ofs.close();
}
