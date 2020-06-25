
#include <iostream>
#include <algorithm>
#include <numeric>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "camFusion.hpp"
#include "dataStructures.h"

using namespace std;


// Create groups of Lidar points whose projection into the camera falls into the same bounding box
void clusterLidarWithROI(std::vector<BoundingBox> &boundingBoxes, std::vector<LidarPoint> &lidarPoints, float shrinkFactor, cv::Mat &P_rect_xx, cv::Mat &R_rect_xx, cv::Mat &RT)
{
    // loop over all Lidar points and associate them to a 2D bounding box
    cv::Mat X(4, 1, cv::DataType<double>::type);
    cv::Mat Y(3, 1, cv::DataType<double>::type);

    for (auto it1 = lidarPoints.begin(); it1 != lidarPoints.end(); ++it1)
    {
        // assemble vector for matrix-vector-multiplication
        X.at<double>(0, 0) = it1->x;
        X.at<double>(1, 0) = it1->y;
        X.at<double>(2, 0) = it1->z;
        X.at<double>(3, 0) = 1;

        // project Lidar point into camera
        Y = P_rect_xx * R_rect_xx * RT * X;
        cv::Point pt;
        pt.x = Y.at<double>(0, 0) / Y.at<double>(0, 2); // pixel coordinates
        pt.y = Y.at<double>(1, 0) / Y.at<double>(0, 2);

        vector<vector<BoundingBox>::iterator> enclosingBoxes; // pointers to all bounding boxes which enclose the current Lidar point
        for (vector<BoundingBox>::iterator it2 = boundingBoxes.begin(); it2 != boundingBoxes.end(); ++it2)
        {
            // shrink current bounding box slightly to avoid having too many outlier points around the edges
            cv::Rect smallerBox;
            smallerBox.x = (*it2).roi.x + shrinkFactor * (*it2).roi.width / 2.0;
            smallerBox.y = (*it2).roi.y + shrinkFactor * (*it2).roi.height / 2.0;
            smallerBox.width = (*it2).roi.width * (1 - shrinkFactor);
            smallerBox.height = (*it2).roi.height * (1 - shrinkFactor);

            // check wether point is within current bounding box
            if (smallerBox.contains(pt))
            {
                enclosingBoxes.push_back(it2);
            }

        } // eof loop over all bounding boxes

        // check wether point has been enclosed by one or by multiple boxes
        if (enclosingBoxes.size() == 1)
        { 
            // add Lidar point to bounding box
            enclosingBoxes[0]->lidarPoints.push_back(*it1);
        }

    } // eof loop over all Lidar points
}


void show3DObjects(std::vector<BoundingBox> &boundingBoxes, cv::Size worldSize, cv::Size imageSize, bool bWait)
{
    // create topview image
    cv::Mat topviewImg(imageSize, CV_8UC3, cv::Scalar(255, 255, 255));

    for(auto it1=boundingBoxes.begin(); it1!=boundingBoxes.end(); ++it1)
    {
        // create randomized color for current 3D object
        cv::RNG rng(it1->boxID);
        cv::Scalar currColor = cv::Scalar(rng.uniform(0,150), rng.uniform(0, 150), rng.uniform(0, 150));

        // plot Lidar points into top view image
        int top=1e8, left=1e8, bottom=0.0, right=0.0; 
        float xwmin=1e8, ywmin=1e8, ywmax=-1e8;
        for (auto it2 = it1->lidarPoints.begin(); it2 != it1->lidarPoints.end(); ++it2)
        {
            // world coordinates
            float xw = (*it2).x; // world position in m with x facing forward from sensor
            float yw = (*it2).y; // world position in m with y facing left from sensor
            xwmin = xwmin<xw ? xwmin : xw;
            ywmin = ywmin<yw ? ywmin : yw;
            ywmax = ywmax>yw ? ywmax : yw;

            // top-view coordinates
            int y = (-xw * imageSize.height / worldSize.height) + imageSize.height;
            int x = (-yw * imageSize.width / worldSize.width) + imageSize.width / 2;

            // find enclosing rectangle
            top = top<y ? top : y;
            left = left<x ? left : x;
            bottom = bottom>y ? bottom : y;
            right = right>x ? right : x;

            // draw individual point
            cv::circle(topviewImg, cv::Point(x, y), 4, currColor, -1);
        }

        // draw enclosing rectangle
        cv::rectangle(topviewImg, cv::Point(left, top), cv::Point(right, bottom),cv::Scalar(0,0,0), 2);

        // augment object with some key data
        char str1[200], str2[200];
        sprintf(str1, "id=%d, #pts=%d", it1->boxID, (int)it1->lidarPoints.size());
        putText(topviewImg, str1, cv::Point2f(left-250, bottom+50), cv::FONT_ITALIC, 2, currColor);
        sprintf(str2, "xmin=%2.2f m, yw=%2.2f m", xwmin, ywmax-ywmin);
        putText(topviewImg, str2, cv::Point2f(left-250, bottom+125), cv::FONT_ITALIC, 2, currColor);  
    }

    // plot distance markers
    float lineSpacing = 2.0; // gap between distance markers
    int nMarkers = floor(worldSize.height / lineSpacing);
    for (size_t i = 0; i < nMarkers; ++i)
    {
        int y = (-(i * lineSpacing) * imageSize.height / worldSize.height) + imageSize.height;
        cv::line(topviewImg, cv::Point(0, y), cv::Point(imageSize.width, y), cv::Scalar(255, 0, 0));
    }

    // display image
    string windowName = "3D Objects";
    cv::namedWindow(windowName, 1);
    cv::imshow(windowName, topviewImg);

    if(bWait)
    {
        cv::waitKey(0); // wait for key to be pressed
    }
}


// associate a given bounding box with the keypoints it contains
void clusterKptMatchesWithROI(BoundingBox &boundingBox, std::vector<cv::KeyPoint> &kptsPrev, std::vector<cv::KeyPoint> &kptsCurr, std::vector<cv::DMatch> &kptMatches) {
  std::vector<cv::DMatch> currMatches;
  std::vector<double> eucliadianDis;
  //find matches and eucliadian distance inside the bounding box
  for(vector<cv::DMatch>::iterator match = kptMatches.begin(); match != kptMatches.end(); ++match) { 
    cv::Point prevPt = kptsPrev[match->queryIdx].pt;
    cv::Point currPt = kptsCurr[match->trainIdx].pt;
    // check wether point is within current bounding box
    cv::Rect smallerBox;//add small margin to eliminate the points that were not on the front car
    double shrinkFactor = 0.15;
    smallerBox.x = boundingBox.roi.x + shrinkFactor * boundingBox.roi.width / 2.0;
    smallerBox.y = boundingBox.roi.y + shrinkFactor * boundingBox.roi.height / 2.0;
    smallerBox.width = boundingBox.roi.width * (1 - shrinkFactor);
    smallerBox.height = boundingBox.roi.height * (1 - shrinkFactor);
    if (smallerBox.contains(currPt)) {
      currMatches.push_back(*match);
      eucliadianDis.push_back(cv::norm(currPt - prevPt));
    }
  }
  double mean = std::accumulate(eucliadianDis.begin(), eucliadianDis.end(), 0.0) / eucliadianDis.size(); //find the average shift of the image
  double threathold = mean*1.5; 
  for(int idx = 0; idx < currMatches.size(); ++idx) {
    if(eucliadianDis[idx] < threathold) { // check if the distance is too far
      boundingBox.keypoints.push_back(kptsCurr[currMatches[idx].trainIdx]); // add the valid key point to the box
      boundingBox.kptMatches.push_back(currMatches[idx]); // add the valid match to the key point
    }
  }
}


// Compute time-to-collision (TTC) based on keypoint correspondences in successive images
void computeTTCCamera(std::vector<cv::KeyPoint> &kptsPrev, std::vector<cv::KeyPoint> &kptsCurr, 
                      std::vector<cv::DMatch> kptMatches, double frameRate, double &TTC, cv::Mat *visImg) {
  // compute distance ratios between all matched keypoints
  vector<double> distRatios; // stores the distance ratios for all keypoints between curr. and prev. frame
  for (auto it1 = kptMatches.begin(); it1 != kptMatches.end() - 1; ++it1) { // outer kpt. loop

    // get current keypoint and its matched partner in the prev. frame
    cv::KeyPoint kpOuterCurr = kptsCurr.at(it1->trainIdx);
    cv::KeyPoint kpOuterPrev = kptsPrev.at(it1->queryIdx);

    for (auto it2 = kptMatches.begin() + 1; it2 != kptMatches.end(); ++it2) { // inner kpt.-loop

      double minDist = 100.0; // min. required distance

      // get next keypoint and its matched partner in the prev. frame
      cv::KeyPoint kpInnerCurr = kptsCurr.at(it2->trainIdx);
      cv::KeyPoint kpInnerPrev = kptsPrev.at(it2->queryIdx);

      // compute distances and distance ratios
      double distCurr = cv::norm(kpOuterCurr.pt - kpInnerCurr.pt);
      double distPrev = cv::norm(kpOuterPrev.pt - kpInnerPrev.pt);
      if (distPrev > std::numeric_limits<double>::epsilon() && distCurr >= minDist) { // avoid division by zero
        double distRatio = distCurr / distPrev;
        distRatios.push_back(distRatio);
      }
    } // eof inner loop over all matched kpts
  }     // eof outer loop over all matched kpts

  // only continue if list of distance ratios is not empty
  if (distRatios.size() == 0) {
    TTC = NAN;
    return;
  }


  // STUDENT TASK (replacement for meanDistRatio)
  std::sort(distRatios.begin(), distRatios.end());
  long medIndex = floor(distRatios.size() / 2.0);
  // compute median dist. ratio to remove outlier influence
  double medDistRatio = distRatios.size() % 2 == 0 ? (distRatios[medIndex - 1] + distRatios[medIndex]) / 2.0 : distRatios[medIndex]; 
  double dT = 1 / frameRate;
  TTC = -dT / (1 - medDistRatio);
  //data validation for FP.6

}


void computeTTCLidar(std::vector<LidarPoint> &lidarPointsPrev,
                     std::vector<LidarPoint> &lidarPointsCurr, double frameRate, double &TTC) {
  // auxiliary variables
  double dT = 1 / frameRate;        // time between two measurements in seconds
  double laneWidth = 4.0; // assumed width of the ego lane
  double reflectionThreathold = 0.5;
  double gapThreathold = 0.1;

  // find closest distance to Lidar points within ego lane
  double minXPrev = 0.00, minXCurr = 0.00;
  std::sort(lidarPointsPrev.begin(), lidarPointsPrev.end(), [&](const LidarPoint x, const LidarPoint y){return x.x < y.x;});
  std::sort(lidarPointsCurr.begin(), lidarPointsCurr.end(), [&](const LidarPoint x, const LidarPoint y){return x.x < y.x;});
  //start from the closest point to check if this point is within specs
  cout << "Lidar Reflection Intensity r: "<<endl;
  int sample_size = 0;//initial ize the sample size for the mean calculation
  for (auto it = lidarPointsPrev.begin(); it != lidarPointsPrev.begin() + 10; ++it) {
    cout << it->r << " "; //data validation for FP.5 to check the refection intensity
    // 3D point within ego lane? not too bright? not a distant outlier?
    if (abs(it->y) <= laneWidth / 2.0 && it->r < reflectionThreathold && abs(it->x - (it+1)->x) < gapThreathold) {
      minXPrev += it->x;
      sample_size++;
    }
  }
  minXPrev = minXPrev/sample_size; //find mean of the first few closest points
  //start from the closest point to check if this point is within specs
  sample_size = 0; //initial ize the sample size for the mean calculation
  for (auto it = lidarPointsCurr.begin(); it != lidarPointsCurr.begin() + 10; ++it) {
    // 3D point within ego lane? not too bright? not a distant outlier?
    if (abs(it->y) <= laneWidth / 2.0 && it->r < reflectionThreathold && abs(it->x - (it+1)->x) < gapThreathold) {
      minXCurr += it->x;
      sample_size++;
    }
  }
  minXCurr = minXCurr/sample_size; //find mean of the first few closest points
  // compute TTC from both measurements
  TTC = minXCurr * dT / (minXPrev - minXCurr);
  
  //data validation for FP.5
  cout<< endl <<"lidarPointsPrev "<< "lidarPointsCurr " <<  "minXPrev " << "minXCurr"<<endl;
  cout << lidarPointsPrev.size() << '\t' << '\t' << lidarPointsCurr.size() << '\t' << "  " <<minXPrev << '\t'<< minXCurr << endl;
  
  // if the distance increased? no valid lidar points? distance did not change? set the ttc to NAN
  if(TTC <= 0 || minXPrev == 0 || minXCurr == 0 || minXPrev == minXCurr) TTC = NAN;
}


void matchBoundingBoxes(std::vector<cv::DMatch> &matches, std::map<int, int> &bbBestMatches, DataFrame &prevFrame, DataFrame &currFrame) {
  std::vector<int> keypointcounts;
  for(vector<BoundingBox>::iterator currBox = currFrame.boundingBoxes.begin(); currBox != currFrame.boundingBoxes.end(); ++currBox) {
    int currPointsCounts = 0; // count the max # of keypoints has been find inside this current box with some previous boxes
    for(vector<BoundingBox>::iterator prevBox = prevFrame.boundingBoxes.begin(); prevBox != prevFrame.boundingBoxes.end(); ++prevBox) {
      int tempPointsCounts = 0; //count the max # of keypoints has been find between this previous box and this current box 
      
      // iterate through all keypoints matches to find the max points inside both boxes
      for(vector<cv::DMatch>::iterator match = matches.begin(); match != matches.end(); ++match) { 
        cv::Point prevPt = prevFrame.keypoints[match->queryIdx].pt;
        cv::Point currPt = currFrame.keypoints[match->trainIdx].pt;
        // check wether point is within current bounding box
        if (currBox->roi.contains(currPt) && prevBox->roi.contains(prevPt)) tempPointsCounts++;
      }
      if(tempPointsCounts > currPointsCounts) { // if better match find between this current box and the previous boxes
        currPointsCounts = tempPointsCounts; // if there is better match update this current box's max # keypoints contained inside
        currBox->trackID = prevBox->boxID; // update this current box best match ID
      }
    }
    keypointcounts.push_back(currPointsCounts);
  }
  /*
  // this print out the total shared points in each current boxes and its previous box
  cout << "currFrame boundingBoxes Size: " << currFrame.boundingBoxes.size() << endl;
  cout << '\t' << "ID" << '\t' << "TrackID  " << "Shared Points" << endl;
  for (int idx = 0; idx < keypointcounts.size(); ++idx) cout << '\t' << idx << '\t' 
    << currFrame.boundingBoxes[idx].trackID << '\t' <<keypointcounts[idx] << '\n'; 
  */
  //check if there is repeats in the matched boxes
  for(vector<BoundingBox>::iterator currBox1 = currFrame.boundingBoxes.begin(); currBox1 != currFrame.boundingBoxes.end(); ++currBox1) {
    vector<BoundingBox>::iterator boxToadd = currBox1;
    for(vector<BoundingBox>::iterator currBox2 = currFrame.boundingBoxes.begin(); currBox2 != currFrame.boundingBoxes.end(); ++currBox2) {
      //if two boxes has the same match with the previous frame, pick the one with the largest amount of the points
      if(boxToadd->trackID == currBox2->trackID && keypointcounts[boxToadd->boxID] < keypointcounts[currBox2->boxID]) boxToadd = currBox2;
    }
    bbBestMatches.insert({ boxToadd->boxID, boxToadd->trackID}); //add this box and its best match to the map
  }
  //assign current bounding box matches to the current frame
  currFrame.bbMatches = bbBestMatches;
}
