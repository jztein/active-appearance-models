#ifndef faceMorpher_facer_h
#define faceMorpher_facer_h

#include <stdio.h>
#include <iostream>
#include <fstream>
#include <stdlib.h>
#include <time.h>

#include "opencv2/core/core.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/nonfree/nonfree.hpp"
#include "opencv2/calib3d/calib3d.hpp"
#include "opencv2/imgproc/imgproc.hpp"

using namespace cv;
using namespace std;

// command line arguments
bool parseArgs(int argc, char** argv);

// Delaunay triangulation
void getFacePointsFromFile(const char* filename, vector<Point2f> &facepoints);
int getFPindex(vector<Point2f> &facepoints, double x, double y);
void delaunay2Faces(Mat im0, Mat im1, vector<vector<Point2f> > &triangles0, vector<vector<Point2f> > &triangles1, vector<Point2f> &facepoints0, vector<Point2f> &facepoints1, string name0, string name1);

// Getting the affine transformations between corresponding triangles
// And getting resultant warped triangle (esp. for intermediate)
Mat getAffine(Point2f src[], Point2f dst[]);
vector<Mat> getAllAffineTransforms(const vector<vector<Point2f> > srcTriangles, const vector<vector<Point2f> > dstTriangles);
void drawAllTriangles(Mat im, const vector<vector<Point2f> > triangles, string name, bool write=false);
vector<vector<Point2f> > getWarpedTriangles(const vector<vector<Point2f> > src, vector<Mat> affines);

// Finding corresponding triangle and thus the affine transformation
// math to check if point is in triangle (trying to find the triangle)
template <class T>
T cross2d(T vecA_x, T vecA_y, T vecB_x, T vecB_y);
int signer(double vecA_x, double vecA_y, double vecB_x, double vecB_y, double vecC_x, double vecC_y);
bool isTriangle(vector<Point2f> triangle, double i, double j);
int getAffineIndex(vector<vector<Point2f> > warpedTriangles, int i, int j);

// Transform image (each triangle_i of image by affine_i)
Mat warpImage(vector<vector<Point2f> > triangles, Mat srcIm, vector<Mat> affine);

// populates vector<vector<vector<Point2f> > > allSrcTriangles, allDstTriangles
void getAllTriangles(vector<vector<Point2f> > srcTriangles, vector<vector<Point2f> > dstTriangles, vector<vector<vector<Point2f> > > &XtoYTriangles);
// Each interpolation set is the affine transform interpolated
void warpAllXLevels(Mat srcIm, vector<vector<Point2f> > srcTriangles, vector<vector<Point2f> > dstTriangles_final, vector<vector<vector<Point2f> > > &XtoYtriangles, vector<Mat> & allXwarpedIms);






#endif
