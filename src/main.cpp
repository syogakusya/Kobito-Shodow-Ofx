#include "ofMain.h"
#include "ofApp.h"

// window size
const int w = 800;
const int h = 600;

int main()
{
	ofSetupOpenGL(w, h, OF_WINDOW);
	ofApp *mainApp = new ofApp;
	mainApp->getWindowSize(w, h);
	ofRunApp(mainApp);
}