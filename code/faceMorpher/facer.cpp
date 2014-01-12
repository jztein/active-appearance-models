#include "facer.h"
#include <dirent.h>

int NUM_STEPS = 20;
double STEP_SIZE = 1./20.;

Scalar RED(0, 0, 255), GREEN(0, 255, 0);
// Don't convert images to gray scale because of this. (images already gray, and want to draw color lines if necessary)

// Image 0's global data
vector<Point2f> facepoints_0;
Mat faceIm_0;
Mat delaunayIm_0;
//vector<Vec6f> triangles_0;
vector<vector<Point2f> > triangles_0;
string imName0;

// Image 1's global data
vector<Point2f> facepoints_1;
Mat faceIm_1;
Mat delaunayIm_1;
//vector<Vec6f> triangles_1;
vector<vector<Point2f> > triangles_1;
string imName1;

// global data
vector<Mat> allAffines;
vector<Mat> allSrcWarpedIms, allDstWarpedIms;
vector<vector<vector<Point2f> > > srcToDstTriangles, dstToSrcTriangles;


void replicateDelaunay(vector<vector<Point2f> > refTriangles, vector<vector<Point2f> > &triangles1, vector<Point2f> facepoints0, vector<Point2f> facepoints1);
void makeEigenfaces(vector<Mat> meanIms, Mat meanOfAll);
void getMeanIm(vector<vector<Mat> > &affinesToMean, vector<Mat> &meanIms, Mat &meanOfAll);

bool EIGEN_ORI = false;
bool EIGEN_WARPED = false;
bool USE_MEAN = false;
vector<Mat> forMeanIms;
//void getMeanIm();
vector<vector<Point2f> > meanFacePoints;

int main(int argc, char** argv)
{
    while (parseArgs(argc-1, argv+1))
        ;
    
    //
    if (USE_MEAN)
    {
        //getMeanIm();
        vector<vector<Mat> > affinesToMean;
        vector<Mat> meanIms;
        Mat meanOfAll;
        getMeanIm(affinesToMean, meanIms, meanOfAll);
        if (EIGEN_WARPED)
            makeEigenfaces(meanIms, meanOfAll); // eigenfaces of warp-to-mean images -> based almost only on appearance
        if (EIGEN_ORI)
            makeEigenfaces(forMeanIms, meanOfAll); // eigenfaces of original images -> shape has a bigger factor
        cout << "end of use_mean" << endl;
        waitKey();
        exit(0);
    }
    
    //
    
    delaunay2Faces(faceIm_0, faceIm_1, triangles_0, triangles_1, facepoints_0, facepoints_1, "delaunay0.jpg", "delaunay1.jpg");
    imwrite("delaunay0.jpg", delaunayIm_0);
    imwrite("delaunay1.jpg", delaunayIm_1);
    cout << "number of triangles: " << triangles_0.size() << endl;
    
    allAffines = getAllAffineTransforms(triangles_0, triangles_1);
    
    warpAllXLevels(faceIm_0.clone(), triangles_0, triangles_1, srcToDstTriangles, allSrcWarpedIms);
    warpAllXLevels(faceIm_1.clone(), triangles_1, triangles_0, dstToSrcTriangles, allDstWarpedIms);
    
    // blend and make images of movie
    //vector<Mat> movieWarps;
    size_t numWarps = allSrcWarpedIms.size();
    for (size_t k = 0; k < numWarps; k++)
    {
        Mat cur = allSrcWarpedIms[k].clone();
        Mat dstIm = allDstWarpedIms[numWarps-k-1];
        for (int i = 0; i < cur.rows; i++)
        {
            for (int j = 0; j < cur.cols; j++)
            {
                cur.at<Vec3b>(i,j) = ((float(numWarps-k))/float(numWarps))* cur.at<Vec3b>(i,j) + ((float(k))/float(numWarps))*dstIm.at<Vec3b>(i,j);
            }
        }
        //movieWarps.push_back(cur);
        
        //n = (k);
        stringstream ss;
        ss << k;
        string ns = ss.str();
        string name(ns+"movie.jpg");
        //imshow(name, cur);
        imwrite(name, cur);
        //waitKey();
    }
    cout << "the end.k" << endl;
    //waitKey();
}


// populates vector<vector<vector<Point2f> > > allSrcTriangles, allDstTriangles
void getAllTriangles(vector<vector<Point2f> > srcTriangles, vector<vector<Point2f> > dstTriangles, vector<vector<vector<Point2f> > > &XtoYTriangles)
{
    XtoYTriangles.clear();
    for (double i = 0; i < 1; i += STEP_SIZE)
    {
        vector< vector<Point2f> > newSetTriangles;
        for (int m = 0; m < srcTriangles.size(); m++)
        {
            vector<Point2f> newTriangle;
            Point2f interS0 = (1-i) * srcTriangles[m][0] + i*dstTriangles[m][0];
            Point2f interS1 = (1-i) * srcTriangles[m][1] + i*dstTriangles[m][1];
            Point2f interS2 = (1-i) * srcTriangles[m][2] + i*dstTriangles[m][2];
            
            newTriangle.push_back(interS0);
            newTriangle.push_back(interS1);
            newTriangle.push_back(interS2);
            
            newSetTriangles.push_back(newTriangle);
        }
        XtoYTriangles.push_back(newSetTriangles);
    }
    // finally, add last point (dst frame)
    XtoYTriangles.push_back(dstTriangles);
}

// each interpolation set is the affine transform interpolated
void warpAllXLevels(Mat srcIm, vector<vector<Point2f> > srcTriangles, vector<vector<Point2f> > dstTriangles_final, vector<vector<vector<Point2f> > > &XtoYtriangles, vector<Mat> & allXwarpedIms)
{
    getAllTriangles(srcTriangles, dstTriangles_final, XtoYtriangles);
    
    cout << "number of interpolation levels: " << XtoYtriangles.size() << endl;
    
    // for one interpolation set of triangles, get warped image
    for (int i = 0; i < XtoYtriangles.size(); i++)
    {
        vector< vector<Point2f> > dstTriangles = XtoYtriangles[i];
        vector<Mat> curAffines = getAllAffineTransforms(srcTriangles, dstTriangles);
        allXwarpedIms.push_back(warpImage(dstTriangles, srcIm, curAffines));
        //imshow("r cur warped", curWarp);
        //waitKey();
        cout << "finished warping level " << i << "..." << endl;
    }
    cout << allXwarpedIms.size() << " : sisze" << endl;
    return;
}



void getMeanIm(vector<vector<Mat> > &affinesToMean, vector<Mat> &meanIms, Mat &meanOfAll)
{
    // get meanTriangles
    // warp all images to meanTriangles mesh
    // mean image = average of warped-to-mean images
    
    // get meanTriangles
    //vector<vector<Mat> > affinesToMean;
    vector<vector<vector<Point2f> > > oriTriangles;
    //vector<Mat> meanIms;
    Mat meanIm;
    Mat im0 = forMeanIms[0].clone();
    Mat im1 = forMeanIms[1];
    vector<vector<Point2f> > triangles0;
    vector<vector<Point2f> > triangles1;
    vector<Point2f> facepoints0 = meanFacePoints[0];
    vector<Point2f> facepoints1 = meanFacePoints[1];
    string name0 = "0"; string name1 = "1";
    delaunay2Faces(im0, im1, triangles0, triangles1, facepoints0, facepoints1, name0, name1);
    
    oriTriangles.push_back(triangles0);
    oriTriangles.push_back(triangles1);
    
    vector<vector<vector<Point2f> > > XtoYTriangles;
    getAllTriangles(triangles0, triangles1, XtoYTriangles);
    
    vector<vector<Point2f> > meanTriangles = XtoYTriangles[XtoYTriangles.size()/2];
//    drawAllTriangles(im0.clone(), meanTriangles, "see triangles");
//    vector<Mat> curAffines = getAllAffineTransforms(triangles0 , meanTriangles);
//    meanIm = warpImage(meanTriangles, im0.clone(), curAffines);
//    imshow("MEAN", meanIm);
//    meanIms.push_back(meanIm);
//    waitKey();
    for (int k = 2; k < forMeanIms.size(); k++)
    {
        im0 = meanIm;
        im1 = forMeanIms[k];
        vector<vector<Point2f> > triangles1;
        vector<Point2f> facepoints1 = meanFacePoints[k];
        replicateDelaunay(triangles0, triangles1, facepoints0, facepoints1);
        
        Mat triangulated = im1.clone();
        stringstream ss;
        ss << k;
        string ns = ss.str();
        string name(ns+"delau.jpg");
        drawAllTriangles(triangulated, triangles1, "cur delaunay replicated");
        imwrite(name, triangulated);
        //waitKey();
        
        oriTriangles.push_back(triangles1);
        
        vector<vector<vector<Point2f> > > XtoYTriangles;
        getAllTriangles(meanTriangles, triangles1, XtoYTriangles);
        
        meanTriangles = XtoYTriangles[XtoYTriangles.size()/2];
        
    }
    // warp all images to mean mesh (meanTriangles)
    for (int k = 0; k < forMeanIms.size(); k++)
    {
        vector<vector<Point2f> > curTriangles = oriTriangles[k];
        vector<Mat> curAffines = getAllAffineTransforms(curTriangles , meanTriangles);
        meanIm = warpImage(meanTriangles, forMeanIms[k].clone(), curAffines);
        stringstream ss;
        ss << k;
        string ns = ss.str();
        string name(ns+"_warpedToMean.jpg");
        cout << "done getting mean from " << k << "..." << endl;
        imshow(name, meanIm);
        imwrite(name, meanIm);
        meanIms.push_back(meanIm);
    }
    
    //Mat meanOfAll = meanIms[0].clone();
    meanOfAll = meanIms[0].clone();
    
    int numMeans = static_cast<int>(meanIms.size());
    for (int i = 0; i < meanOfAll.rows; i++)
    {
        for (int j =0; j < meanOfAll.cols; j++)
        {
            int B = 0, G = 0, R = 0;
            for (int k = 0; k < numMeans; k++)
            {
                B += meanIms[k].at<Vec3b>(i,j)[0];
                G += meanIms[k].at<Vec3b>(i,j)[1];
                R += meanIms[k].at<Vec3b>(i,j)[2];
            }
            B /= numMeans;
            if (B > 255) B = 255; // clamp
            G /= numMeans;
            if (G > 255) G = 255;
            R /= numMeans;
            if (R > 255) R = 255;
            
            meanOfAll.at<Vec3b>(i,j) = Vec3b(B, G, R);
        }
    }
    imshow("MEAN of all MEANS", meanOfAll);
    imwrite("MEAN_img.jpg", meanOfAll);
    cout << "Finished getting mean. Press ANY key to continue..." << endl;
    waitKey();
}

void makeEigenfaces(vector<Mat> meanIms, Mat meanOfAll)
{
    // since it is a gray image, just need to consider one channel (since R = G = B for gray)
    
    // 1. Do PCA (principle component analysis) by computing eigenfaces:
    
    // - get difference of warped-to-mean individual images from mean image
    // - put into one long vector, getting M number of long vectors (M is number of images)
    // - Let A = (R*C)xM-matrix, where R*C is row * col size of the image
    // - compute covariance matrix C by A' * A (getting MxM-matrix), and calculate the eigenvectors and eigenvalues
    // - reverse eigenvectors from one long vector back to RxC-matrix, giving eigenface
    
    int numMeanIms = static_cast<int>(meanIms.size());
    
    cout << "numMeanIms: " << numMeanIms << endl;
    int R = meanIms[0].rows, C = meanIms[0].cols;
    int RC = R*C;
    
    Mat A(RC, numMeanIms, CV_32F, Scalar(0));
    // Cov = Covariance matrix
    // Cov = A'*A, where A is made up of the long vectors (not A*A' since that matrix will be too big)
    Mat Cov;
    
    int i = 0, j = 0, d = 0;
    for (int k = 0; k < numMeanIms; k++)
    {
        Mat cur = meanIms[k];
        Mat diff = meanOfAll - cur;
        stringstream ss;
        ss << k;
        string ns = ss.str();
        string name(ns+"_diff.jpg");
        imshow(name, diff);
        imwrite(name, diff);
        for (int r = 0; r < RC; r++)
        {
            i = r/C;
            j = r - i*C;
            d = diff.at<Vec3b>(i,j)[0];
            A.at<float>(r, k) = d;
        }
    }

    Mat At = A.t();
    
    cout << "A:r,c: " << A.rows << ", " << A.cols << endl;
    cout << "At:r,c: " << At.rows << ", " << At.cols << endl;
    
    Cov = At * A;
    //cout << Cov << endl;
    //waitKey();
    Mat eigenvalues, eigenvectors;
    eigen(Cov, eigenvalues, eigenvectors);
    
    // eigen() gives back the eigenvectors in this order: from largest eigenvalue to smallest eigenvalue
    // eigenvectors should be normalized, and in range -1 to 1
    cout << "EIGENVECTORS\n" << eigenvectors << "\nEIGENVALUES:\n" << eigenvalues << endl;
    //waitKey();
    
    // we have the eigenvectors of A'*A.
    // to get the eigenvectors of A*A', which are the eigenfaces,
    // A * eigenVector
    Mat allEigenfaces = A * eigenvectors.t(); // eigenvectors.t() for opencv's sake (eigenvectors in rows)
    
    Mat eigenBoss = allEigenfaces;
    
    // recover eigenfaces
    vector<Mat> eigenfaces;
    
    Mat blank = meanIms[0].clone();
    for (int i = 0; i < blank.rows; i++)
        for (int j = 0; j < blank.cols; j++)
            blank.at<Vec3b>(i,j) = Vec3b(0,0,0);
    
    for (int k = 0; k < numMeanIms; k++)
    {
        Mat cur = blank.clone();
        for (int r = 0; r < RC; r++)
        {
            i = r/C;
            j = r - i*C;
            int c = 0;
            float t = eigenBoss.at<float>(r,k);
            if (t != 0.)
                c = (t+1.)*255./2.;
            if (c > 255) c = 255; // clamp values
            if (c < 0)   c = 0;
            cur.at<Vec3b>(i, j) = Vec3b(c,c,c);
        }
        eigenfaces.push_back(cur);
    }
    
    // show and write all eigenfaces
    for (int k = 0; k < eigenfaces.size(); k++)
    {
        stringstream ss;
        ss << k;
        string ns = ss.str();
        string name(ns+"_ef.jpg");
        imshow(name, eigenfaces[k]);
        imwrite(name, eigenfaces[k]);
    }
    cout << "Finished making eigenfaces. Press ANY key to continue... " << endl;
    waitKey();

}

void replicateDelaunay(vector<vector<Point2f> > refTriangles, vector<vector<Point2f> > &triangles1, vector<Point2f> facepoints0, vector<Point2f> facepoints1)
{
    
    vector<Point2f> triVertices2(3); // 1 triangle has 3 vertices
    vector<Point2f> triangle;
    Mat delaunayIm1;
    for (int k = 0; k < refTriangles.size(); k++)
    {
        triangle = refTriangles[k];
        
        // find corresponding vertices/ triangle
        int vert1idx = getFPindex(facepoints0, triangle[0].x, triangle[0].y);
        int vert2idx = getFPindex(facepoints0, triangle[1].x, triangle[1].y);
        int vert3idx = getFPindex(facepoints0, triangle[2].x, triangle[2].y);
        triVertices2[0].x = facepoints1[vert1idx].x;
        triVertices2[0].y = facepoints1[vert1idx].y;
        triVertices2[1].x = facepoints1[vert2idx].x;
        triVertices2[1].y = facepoints1[vert2idx].y;
        triVertices2[2].x = facepoints1[vert3idx].x;
        triVertices2[2].y = facepoints1[vert3idx].y;
        
        // update trianglelists with acceptable triangles
        triangles1.push_back(triVertices2);
        
        // draw face1's triangles
        line(delaunayIm1, Point(triVertices2[0].x, triVertices2[0].y),
             Point(triVertices2[1].x, triVertices2[1].y), GREEN);
        line(delaunayIm1, Point(triVertices2[1].x, triVertices2[1].y),
             Point(triVertices2[2].x, triVertices2[2].y), GREEN);
        line(delaunayIm1, Point(triVertices2[2].x, triVertices2[2].y),
             Point(triVertices2[0].x, triVertices2[0].y), GREEN);
    }

}


// NOTE: triangles vertices must be user-chosen in the same orientation and index
//***02centerlight.jpg
// parse command line arguments for image 0 and image 1's global data
bool parseArgs(int argc, char** argv)
{
    if (argc < 2) // arguments come in pairs
    {
        // usage
        cout << "Usage:\nMorphing one face to another\n./faceMorpher -i0 <image0.jpg> -f0 <landmarks0>.txt -i1 <image1.jpg> -f1 <landmarks1.txt>\n"
        << "Getting mean image, and all individual images warped to mean\n./faceMorpher -m ./<mean image dir>/ -mf ./<mean landmarks dir>/\n"
        << "Gets eigenfaces. Does mean getting too.\n./faceMorpher -m ./<mean image dir>/ -mf ./<mean landmarks dir>/ -e <1|0>"
        << "\n -e 1 for eigenfaces of warped-to-mean images; 0 for eigenfaces of original images"
        << endl;
        return false;
    }
    
    // still have arguments to parse
    // option key
    char* curArg = argv[0];
    
    // face image 0
    if (strcmp(curArg, "-i0") == 0)
    {
        faceIm_0 = imread(argv[1]);
        if (!faceIm_0.data)
        {
            cout << "Cannot read target file" << endl;
            return false;
        }
        imName0 = string(argv[1]);
        parseArgs(argc-2, argv+2);
    }
    // get facepoints file and parse it
    else if (strcmp(curArg, "-f0") == 0)
    {
        getFacePointsFromFile(argv[1], facepoints_0);
        cout << "facepoints0 read: " << facepoints_0.size() << endl;
        parseArgs(argc-2, argv+2);
    }
    
    // face image 1
    else if (strcmp(curArg, "-i1") == 0)
    {
        faceIm_1 = imread(argv[1]);
        if (!faceIm_1.data)
        {
            cout << "Cannot read target file" << endl;
            return false;
        }
        imName1 = string(argv[1]);
        parseArgs(argc-2, argv+2);
    }
    else if (strcmp(curArg, "-f1") == 0)
    {
        getFacePointsFromFile(argv[1], facepoints_1);
        cout << "facepoints1 read: " << facepoints_1.size() << endl;
        parseArgs(argc-2, argv+2);
    }
    // load color images for mean
    else if (strcmp(curArg, "-m") == 0)
    {
        USE_MEAN = true;
        
        const char* bluesDir = argv[1];
        const string dir(bluesDir);
        
        // get all images from folder (same dir as executable)
        DIR* dirp = opendir(bluesDir);
        dirent *dp;
        if (!dirp)
        {
            cout << "For Mean Color images directory invalid" << endl;
            return false;
        }
        while ((dp = readdir(dirp)) != NULL) // hits NULL on end of directory
        {
            if (dp->d_name[0] != '.' )//|| dp->d_name[0] )
            {
                Mat curColor = Mat(imread((dir+dp->d_name).c_str()));
                if (!curColor.data)
                {
                    cout << (dir+dp->d_name).c_str() << " not readable" << endl;
                    return false;
                }
                forMeanIms.push_back(curColor);
            }
        }
        closedir(dirp);
        
        cout << "num forMean images read: " <<  forMeanIms.size() << endl;

        parseArgs(argc-2, argv+2);
        return false;
    }
    else if (strcmp(curArg, "-mf") == 0)
    {
        
        const char* bluesDir = argv[1];
        const string dir(bluesDir);
        
        // get all images from folder (same dir as executable)
        DIR* dirp = opendir(bluesDir);
        dirent *dp;
        if (!dirp)
        {
            cout << "For Mean landmarks directory invalid" << endl;
            return false;
        }
        
        while ((dp = readdir(dirp)) != NULL) // hits NULL on end of directory
        {
            if (dp->d_name[0] != '.' )//|| dp->d_name[0] )
            {
                vector<Point2f> newFacePoints;
                const char* name = ((dir+dp->d_name).c_str());
                //cout << name << ": ";
                getFacePointsFromFile(name, newFacePoints);
                //cout << newFacePoints[5] << endl;
                meanFacePoints.push_back(newFacePoints);
            }
        }
        closedir(dirp);
        
        cout << "num forMean facepoints read: " <<  meanFacePoints.size() << endl;
        
        
        parseArgs(argc-2, argv+2);
        return false;
    }
    // to get eigen faces or not (will have to do USE_MEAN too)
    else if (strcmp(curArg, "-e") == 0)
    {
        int flag = atoi(argv[1]);
        if (flag == 1)
            EIGEN_WARPED = true;
        else if (flag == 0)
            EIGEN_ORI = true;
        parseArgs(argc-2, argv+2);
        return false;
    }

    
    return false;
}

// Parse file to get manually chosen face landmarks, facepoints
void getFacePointsFromFile(const char* filename, vector<Point2f> &facepoints)
{
    string line;
    ifstream facePointsFile(filename);
    if (facePointsFile.is_open())
    {
        while(facePointsFile.good())
        {
            getline(facePointsFile, line);
            
            double xs[2] = {0., 0.};
            stringstream ss(line);
            string x;
            size_t k = 0;
            while(getline(ss, x, ' '))
            {
                xs[k] = atof(x.c_str())/2; // <------ HACK around facepointChooser.py which blows up face image by 2.
                k++;
            }
            
            if (x.size())
            {
                facepoints.push_back(Point2f(xs[0], xs[1]));
            }
        }
        facePointsFile.close();
    }
    else
    {
        cout << "Unable to open " << facePointsFile << endl;
    }
}

const double FPidx_EPSILON = .1;
int getFPindex(vector<Point2f> &facepoints, double x, double y)
{
    for (int k = 0; k < facepoints.size(); k++)
    {
        if (facepoints[k].x <= x+FPidx_EPSILON && facepoints[k].y <= y + FPidx_EPSILON
            && facepoints[k].x >= x-FPidx_EPSILON && facepoints[k].y >= y - FPidx_EPSILON)
        {
            return k;
        }
    }
    // should not reach here
    cout << "getFPindex error: couldn't find facepoint index" << endl;

    return 0;
}

// get same delaunay triangulation for two facesand draw the triangles
void delaunay2Faces(Mat im0, Mat im1, vector<vector<Point2f> > &triangles0, vector<vector<Point2f> > &triangles1, vector<Point2f> &facepoints0, vector<Point2f> &facepoints1, string name0, string name1)
{
    Rect imFrame(0, 0, im0.cols, im0.rows);
    cout << "imFrame: " << im0.cols << ", " << im0.rows << endl;
    Subdiv2D subdiv2d(imFrame);
    
    // Want to do random insertion of points to be faster -- verify this?
    // So need to keep track of which points have be inserted since we have a fixed set of points to insert
    size_t numFacePoints = facepoints0.size();
    vector<bool> inserted(numFacePoints, false);
    size_t count = 0;
    while(count < numFacePoints)
    {
        size_t idx = count;//rand() % numFacePoints;
        
        // if inserted already, do not re-insert (search for other points)
        if (inserted[idx])
            continue;
        
        inserted[idx] = true;
        
        // else, we can add this point
        count++;
        
        Point2f curPt = facepoints0[idx];
        
        /// locate_point
        int edge0=0, vertex=0;
        subdiv2d.locate(curPt, edge0, vertex);
        
        if(edge0 > 0)
        {
            int edge = edge0;
            do
            {
                Point2f org, dst;
                if( subdiv2d.edgeOrg(edge, &org) > 0 && subdiv2d.edgeDst(edge, &dst) > 0 )
                    ;//line( delaunayIm, org, dst, RED, 1, 1, 0 );
                
                edge = subdiv2d.getEdge(edge, Subdiv2D::NEXT_AROUND_LEFT);
            }
            while(edge != edge0);
        }
        ///
        
        subdiv2d.insert(curPt);
        
        // else triangles will just draw on top of each other
        Mat delaunayIm0 = im0.clone();
        Mat delaunayIm1 = im1.clone();
        
        //drawDelaunayTriangles(im, subdiv2d, RED);
        vector<Vec6f> newTriangles0;
        subdiv2d.getTriangleList(newTriangles0);
        vector<Point2f> triVertices(3), triVertices2(3); // 1 triangle has 3 vertices
        Vec6f triangle;
        
        // clear triangle lists as we are re-populating them
        triangles0.clear();
        triangles1.clear();
        
        for (int k = 0; k < newTriangles0.size(); k++)
        {
            triangle = newTriangles0[k];
            
            // don't accept bad triangles (for some reason getTriangleList generates extra out-of-bound triangles)
            if (triangle[0] >= im0.cols || triangle[2] >= im0.cols || triangle[4] >= im0.cols || triangle[0] < 0 || triangle[2] < 0 || triangle[4] < 0 || triangle[5] < 0 || triangle[1] < 0 || triangle[3] < 0 || triangle[1] > im0.rows || triangle[3] > im0.rows || triangle[5] > im0.rows)
                continue;
            
            // populate vertices of one triangle
            triVertices[0].x = triangle[0];//cvRound(triangle[0]);
            triVertices[0].y = triangle[1];//cvRound(triangle[1]);
            triVertices[1].x = triangle[2];//cvRound(triangle[2]);
            triVertices[1].y = triangle[3];//cvRound(triangle[3]);
            triVertices[2].x = triangle[4];//cvRound(triangle[4]);
            triVertices[2].y = triangle[5];//cvRound(triangle[5]);
            
            // find corresponding vertices/ triangle
            int vert1idx = getFPindex(facepoints0, triangle[0], triangle[1]);
            int vert2idx = getFPindex(facepoints0, triangle[2], triangle[3]);
            int vert3idx = getFPindex(facepoints0, triangle[4], triangle[5]);
            triVertices2[0].x = facepoints1[vert1idx].x;
            triVertices2[0].y = facepoints1[vert1idx].y;
            triVertices2[1].x = facepoints1[vert2idx].x;
            triVertices2[1].y = facepoints1[vert2idx].y;
            triVertices2[2].x = facepoints1[vert3idx].x;
            triVertices2[2].y = facepoints1[vert3idx].y;
            
            // update trianglelists with acceptable triangles
            triangles0.push_back(triVertices);
            triangles1.push_back(triVertices2);
            
            // draw face0's triangles
            line(delaunayIm0, Point(triVertices[0].x, triVertices[0].y),
                 Point(triVertices[1].x, triVertices[1].y), GREEN);
            line(delaunayIm0, Point(triVertices[1].x, triVertices[1].y),
                 Point(triVertices[2].x, triVertices[2].y), GREEN);
            line(delaunayIm0, Point(triVertices[2].x, triVertices[2].y),
                 Point(triVertices[0].x, triVertices[0].y), GREEN);
            
            // draw face1's triangles
            line(delaunayIm1, Point(triVertices2[0].x, triVertices2[0].y),
                 Point(triVertices2[1].x, triVertices2[1].y), GREEN);
            line(delaunayIm1, Point(triVertices2[1].x, triVertices2[1].y),
                 Point(triVertices2[2].x, triVertices2[2].y), GREEN);
            line(delaunayIm1, Point(triVertices2[2].x, triVertices2[2].y),
                 Point(triVertices2[0].x, triVertices2[0].y), GREEN);
        }
        delaunayIm_0 = delaunayIm0.clone();
        delaunayIm_1 = delaunayIm1.clone();
    }
}

vector<Mat> getAllAffineTransforms(const vector<vector<Point2f> > srcTriangles, const vector<vector<Point2f> > dstTriangles)
{
    vector<Mat> curSetAffines;
    for (int i = 0; i < srcTriangles.size(); i++)
    {
        Point2f vf0 = srcTriangles[i][0];
        Point2f vf1 = srcTriangles[i][1];
        Point2f vf2 = srcTriangles[i][2];
        Point2f src[3] = {vf0, vf1, vf2};
        Point2f vt0 = dstTriangles[i][0];
        Point2f vt1 = dstTriangles[i][1];
        Point2f vt2 = dstTriangles[i][2];
        Point2f dst[3] = {vt0, vt1, vt2};
        Mat Affine = getAffineTransform(src, dst);
        /*
        // last row of Affine mat: [0 0 1]
        double* lastAffineMatRow = (double*)Affine.data;
        
        for (int k = 0; k < 6; k++)
        {
            if (lastAffineMatRow[k] < 0.0001 && lastAffineMatRow[k] > -0.0001)
                lastAffineMatRow[k] = 0.;
            if (lastAffineMatRow[k] < 1.0001 && lastAffineMatRow[k] > .9999)
                lastAffineMatRow[k] = 1.;
        }*/

        curSetAffines.push_back(Affine);//getAffineTransform(src, dst));
    }
    
    return curSetAffines;
}

void drawAllTriangles(Mat im, const vector<vector<Point2f> > triangles, string name, bool write)
{
    // draw all triangles
    vector<Point2f> triangle(3);
    vector<Point2d> triVertices(3);
    for (int k = 0; k < triangles.size(); k++)
    {
        triangle = triangles[k];
        
        triVertices[0].x = cvRound(triangle[0].x);
        triVertices[0].y = cvRound(triangle[0].y);
        triVertices[1].x = cvRound(triangle[1].x);
        triVertices[1].y = cvRound(triangle[1].y);
        triVertices[2].x = cvRound(triangle[2].x);
        triVertices[2].y = cvRound(triangle[2].y);
        
        // draw triangle
        line(im, Point(triVertices[0].x, triVertices[0].y),
             Point(triVertices[2].x, triVertices[2].y), RED);
        
        line(im, Point(triVertices[0].x, triVertices[0].y),
             Point(triVertices[1].x, triVertices[1].y), RED);
        
        line(im, Point(triVertices[1].x, triVertices[1].y),
             Point(triVertices[2].x, triVertices[2].y), RED);
    }
    imshow(name, im);
    if (write)
        imwrite(name, im);
}

vector<vector<Point2f> > getWarpedTriangles(const vector<vector<Point2f> > src, vector<Mat> affines)
{
    vector<vector<Point2f> > warpedTriangles;
    
    assert(affines.size() == src.size());
    
    for (int k = 0; k < src.size(); k++)
    {
        Mat affine = affines[k];
        vector<Point2f> cur = src[k];
        vector<Point2f> transformed(3);
        for (int i = 0; i < cur.size(); i++)
        {
            Point2f tP;
            tP.x =  affine.at<double>(0,0)*cur[i].x +
                    affine.at<double>(0,1)*cur[i].y +
                    affine.at<double>(0,2);
            if (tP.x < 0.)
                tP.x = 0.;
            tP.y =  affine.at<double>(1,0)*cur[i].x +
                    affine.at<double>(1,1)*cur[i].y +
                    affine.at<double>(1,2);
            if (tP.y < 0.)
                tP.y = 0.;
            transformed[i] = tP;
        }
        warpedTriangles.push_back(transformed);
    }
    assert(affines.size() == warpedTriangles.size());
    
    return warpedTriangles;
}

// return magnitude of cross product
template <class T>
T cross2d(T vecA_x, T vecA_y, T vecB_x, T vecB_y)
{
    T result = vecA_x * vecB_y - vecA_y * vecB_x;
    //if (result < 0) // abs
    //    return -result;
    return result;
}

// returns: <vecA x vecB, vecA x vecC> >= 0
int signer(double vecA_x, double vecA_y, double vecB_x, double vecB_y, double vecC_x, double vecC_y)
{
    double sign = (vecA_x * vecB_y - vecA_y * vecB_x)*(vecA_x * vecC_y - vecA_y * vecC_x);
    if (sign < 0)
        return -1;
    else if (sign >= -0.0001 && sign <= 0.0001)
        return 0;
    return 1;
}

double tEPSILON = 0.01;

// check via cross products around A, B, and C with other vertices and Point
// if Point lies in the triangle, the cross products (in a fixed right-hand rule orientation) should have the same sign
bool isTriangle(vector<Point2f> triangle, double i, double j)
{
    Point2f AB = triangle[1]-triangle[0];
    Point2f AP = Point2f(i,j) - triangle[0];
    Point2f BC = triangle[2] - triangle[1];
    Point2f BP = Point2f(i,j) - triangle[1];
    Point2f CA = triangle[0] - triangle[2];
    Point2f CP = Point2f(i,j) - triangle[2];
    
    int sABAP_ABAC = signer(AB.x, AB.y, AP.x, AP.y, -CA.x, -CA.y);
    int sBCBP_BCBA = signer(BC.x, BC.y, BP.x, BP.y, -AB.x, -AB.y);
    int sCACP_CACB = signer(CA.x, CA.y, CP.x, CP.y, -BC.x, -BC.y);
    
    if (sABAP_ABAC >= 0 && sBCBP_BCBA >= 0 && sCACP_CACB >= 0)
        return true;
    
    double crossABAP = cross2d(AB.x, AB.y, AP.x, AP.y);
    double crossBCBP = cross2d(BC.x, BC.y, BP.x, BP.y);
    double crossCACP = cross2d(CA.x, CA.y, CP.x, CP.y);
    
    // if two cross products are 0, inside
    if (((crossABAP >= -tEPSILON && crossABAP <= tEPSILON)
          && (crossBCBP >= -tEPSILON && crossBCBP <= tEPSILON)) ||
        ((crossABAP >= -tEPSILON && crossABAP <= tEPSILON)
         && (crossCACP >= -tEPSILON && crossCACP <= tEPSILON)) ||
        ((crossCACP >= -tEPSILON && crossCACP <= tEPSILON)
         && (crossBCBP >= -tEPSILON && crossBCBP <= tEPSILON)))
        return true;
    
    // if one cross product is 0, the other two equal each other, inside
    if (((crossABAP >= -tEPSILON && crossABAP <= tEPSILON) &&
         (crossBCBP <= crossCACP + tEPSILON && crossBCBP >= crossCACP-tEPSILON)) ||
        ((crossBCBP >= -tEPSILON && crossBCBP <= tEPSILON) &&
         (crossABAP <= crossCACP + tEPSILON && crossABAP >= crossCACP-tEPSILON)) ||
        ((crossCACP >= -tEPSILON && crossCACP <= tEPSILON) &&
         (crossBCBP <= crossABAP + tEPSILON && crossBCBP >= crossABAP-tEPSILON)))
        return true;
    
    //if (i <= faceIm_0.rows || j <= faceIm_0.cols)
    //    return true;
    
    return false;
}

int getAffineIndex(vector<vector<Point2f> > warpedTriangles, int i, int j)
{
    for (int k = 0; k < warpedTriangles.size(); k++)
    {
        if (isTriangle(warpedTriangles[k], double(i), double(j)))
        {
            /* visually check if triangle picking is correct
            if (i % 50 == 0 && j % 50 == 0)
            {
                vector<Point2f> curTriangle = warpedTriangles[k];
                 Mat hey = faceIm_0.clone();
                 line(hey, Point(i,j), Point(i+1,j+1), RED, 5);
                 
                 line(hey, Point(curTriangle[0].x, curTriangle[0].y),
                 Point(curTriangle[2].x, curTriangle[2].y), GREEN, 2);
                 
                 line(hey, Point(curTriangle[0].x, curTriangle[0].y),
                 Point(curTriangle[1].x, curTriangle[1].y), GREEN, 2);
                 
                 line(hey, Point(curTriangle[1].x, curTriangle[1].y),
                 Point(curTriangle[2].x, curTriangle[2].y), GREEN, 2);
                 imshow("hello", hey);
                 waitKey();
            }//*/
            return k;
        }
    }
    return -1;
}

// triangles = warped triangles
Mat warpImage(vector<vector<Point2f> > triangles, Mat srcIm, vector<Mat> affines)
{
    Mat warpedIm = srcIm.clone();
    //*
     for (int i = 0; i < warpedIm.rows; i++)
        for (int j = 0; j < warpedIm.cols; j++)
            warpedIm.at<Vec3b>(i,j) = Vec3b(0,0,0);
    //*/
    
    for (int i = 0; i < warpedIm.cols; i++)//
    {
        for (int j = 0; j < warpedIm.rows; j++)
        {
            // get warped triangle belonged to
            int idx = getAffineIndex(triangles, i,j);
            if (idx == -1)
            {
                //cout << "bad triangle index found" << endl;
                continue;
            }
            
            Mat curA = affines[idx];
            
            //*
            // affine mat: [ R     | T ] inv: [ R^-1  | -R^-1 T]
            //             [ 0 0 0 | 1 ] inv: [ 0 0 0 | 1      ]
            Mat R(2, 2, CV_64F); // NOT JUST ROTATION: rotation, scaling, shear
            R.at<double>(0,0) = curA.at<double>(0,0);
            R.at<double>(0,1) = curA.at<double>(0,1);
            R.at<double>(1,0) = curA.at<double>(1,0);
            R.at<double>(1,1) = curA.at<double>(1,1);
            Mat invR = R.inv(DECOMP_SVD);
            Mat invA(3, 3, CV_64F, 0.); // for (3,0-2)
            // R^-1
            invA.at<double>(0,0) = invR.at<double>(0,0);
            invA.at<double>(0,1) = invR.at<double>(0,1);
            invA.at<double>(1,0) = invR.at<double>(1,0);
            invA.at<double>(1,1) = invR.at<double>(1,1);
            // -R^-1 T
            invA.at<double>(0,2) = -invR.at<double>(0,0)* curA.at<double>(0,2) -invR.at<double>(0,1)* curA.at<double>(1,2);
            invA.at<double>(1,2) = -invR.at<double>(1,0)* curA.at<double>(0,2) -invR.at<double>(1,1)* curA.at<double>(1,2);
            // 1
            //invA.at<double>(2,2) = 1.;
            
            //*/
            //Mat invA = curA.inv(DECOMP_SVD);
            //
            double x(i), y(j);
            
            double srcJ = invA.at<double>(0,0)*x + invA.at<double>(0,1)*y + invA.at<double>(0,2);
            double srcI = invA.at<double>(1,0)*x + invA.at<double>(1,1)*y + invA.at<double>(1,2);
            
            if (srcI < 0 || srcJ < 0)
                warpedIm.at<Vec3b>(j, i) =Vec3b(0,0,0);//srcIm.at<Vec3b>(j,i);//i, j); // hack
            else
                warpedIm.at<Vec3b>(j, i) = srcIm.at<Vec3b>(srcI,srcJ);
        }
    }
    
    //drawAllTriangles(warpedIm.clone(), triangles, "post warp", false);
    //waitKey();
    return warpedIm;
}


/*/ checking triangles e.g. put after first few lines of warpImage()
 Mat mem = warpedIm.clone();
 Point2f triVertices[3];
 for (int k =0; k < triangles.size(); k++)
 {
 
 vector<Point2f> curTriangle = triangles[k];
 
 triVertices[0].x = (curTriangle[0].x);
 triVertices[0].y = (curTriangle[0].y);
 triVertices[1].x = (curTriangle[1].x);
 triVertices[1].y = (curTriangle[1].y);
 triVertices[2].x = (curTriangle[2].x);
 triVertices[2].y = (curTriangle[2].y);
 
 Scalar COLOR = Scalar(rand()%255, rand()%255, rand()%255);
 line(mem, Point(triVertices[0].x, triVertices[0].y),
 Point(triVertices[2].x, triVertices[2].y), COLOR, 3);
 
 line(mem, Point(triVertices[0].x, triVertices[0].y),
 Point(triVertices[1].x, triVertices[1].y), COLOR, 3);
 
 line(mem, Point(triVertices[1].x, triVertices[1].y),
 Point(triVertices[2].x, triVertices[2].y), COLOR, 3);
 
 imshow("what triangle?", mem);
 waitKey();
 } //*/



