#include <stdio.h>
#include <math.h>
#include <iostream>
#include "opencv2/core.hpp"
#include "opencv2/features2d.hpp"
#include "opencv2/calib3d.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/viz.hpp"
#include "opencv2/xfeatures2d.hpp"
#include <opencv2/imgproc/imgproc.hpp>
#include "displaypointline.h"
#include "feature_match.h"
#include "findfundamental.h"

using namespace std;
using namespace cv;
using namespace cv::xfeatures2d;

#define use_SIFT 0
void rotation(Mat F,Mat &R,Mat &T,int r1);


Mat cameraMatrix();

void show_keypoints(String title, 
                    Mat img_1, 
                    std::vector<KeyPoint> keypoints_1, 
                    Mat img_2,
                    std::vector<KeyPoint> keypoints_2
                    );


void show_matches(String title,
                  Mat img_1, 
                  std::vector<KeyPoint> keypoints_1, 
                  Mat img_2,
                  std::vector<KeyPoint> keypoints_2,  
                  std::vector< DMatch > Gmatches
                  );

std::vector< DMatch > test( Mat img_1,
                            vector<KeyPoint> m_LeftKey,
                            Mat img_2,
                            vector<KeyPoint> m_RightKey,
                            vector<DMatch> m_Matches
                          );

void PLYfrom2D( String file,
                Mat img_1,
                vector<KeyPoint> m_LeftKey,
                Mat img_2,
                vector<KeyPoint> m_RightKey,
                vector<DMatch> m_Matches
);

void fundaemtaleachpair(string s1, string s2)
{ 
  string i1 = s1;
  string i2 = s2;

  Mat img_1 = imread( i1+".png", IMREAD_GRAYSCALE );
  Mat img_2 = imread( i2+".png", IMREAD_GRAYSCALE );


  //  Task 1.1.Feature Detection. 
  //  Detect image features using SIFT, SURF.
  //   Compute the corresponding descriptors and visualize the features
  //-- Step 1: Detect the keypoints using SURF Detector, compute the descriptors
    cout<<endl<<" ---Task 1.1.Feature Detection---"<<endl;
  ////Hessian matrix describes the second derivatives of a function
  //  the product of eigenvalues is the determinant of Hessian. 
  //  minHessian here could be consider as
  //   that determinant, which is how "sharp" the extrema you need.
  //   If one point's Det(Hessian) is larger than that value, it could be the interest point.
  int minHessian = 1400; 
  //create	(	double 	hessianThreshold = 100,
  // int 	nOctaves = 4,
  // int 	nOctaveLayers = 3,
  // bool 	extended = false,
  // bool 	upright = false 
  // )	
  Ptr<SURF> detector = SURF::create();
  //set new min hessian 
  detector->setHessianThreshold(minHessian);
  //turn ture to get 128 vector space descriptors rather than 64
  detector->setExtended (true);

  std::vector<KeyPoint> surf_keypoints_1, surf_keypoints_2;
  Mat surf_descriptors_1, surf_descriptors_2;

  detector->detectAndCompute( img_1, Mat(), surf_keypoints_1, surf_descriptors_1 );
  detector->detectAndCompute( img_2, Mat(), surf_keypoints_2, surf_descriptors_2 );
  cout<<"Using SURF"<<endl;  
  cout<<"Press any button on imshow window to proceed"<<endl<<endl;
  show_keypoints("surf ", img_1, surf_keypoints_1, img_2, surf_keypoints_2 );

  Ptr<SIFT> detector2 = SIFT::create();
  std::vector<KeyPoint> sift_keypoints_1, sift_keypoints_2;
  Mat sift_descriptors_1, sift_descriptors_2;
  detector2->detectAndCompute( img_1, Mat(), sift_keypoints_1, sift_descriptors_1 );
  detector2->detectAndCompute( img_2, Mat(), sift_keypoints_2, sift_descriptors_2 );
  cout<<"Using SIFT"<<endl;
  cout<<"Press any button on imshow window to proceed"<<endl<<endl;
  show_keypoints("sift ", img_1, sift_keypoints_1, img_2, sift_keypoints_2 );
  
  std::vector<KeyPoint> keypoints_1, keypoints_2;
  Mat descriptors_1, descriptors_2;
  //-- Step 2: Matching descriptor vectors with a brute force matcher
  if(use_SIFT == 1)
  {
    keypoints_1 = sift_keypoints_1, keypoints_2 = sift_keypoints_2;
     descriptors_1 = sift_descriptors_1 , descriptors_2 = sift_descriptors_2;
    cout<<"use SIFT to futher calculate the Feature Matching"<<endl;
  }
  else{
    keypoints_1 = surf_keypoints_1, keypoints_2 = surf_keypoints_2;
    descriptors_1 = surf_descriptors_1 , descriptors_2 = surf_descriptors_2;
    cout<<"use SURF to futher calculate the Feature Matching"<<endl;
  }

  // cout << "Image1 = "<< endl << " "  << descriptors_1.size() << endl << endl;
  // cout << "Image2 = "<< endl << " "  << descriptors_2.size() << endl << endl;

  cout<<endl<<" ---Task 1.2. Feature Matching---"<<endl;
  std::vector< DMatch > Gmatches1;
  std::vector< DMatch > Gmatches2;
  std::vector< DMatch > Best_matches;
  
  std::vector< DMatch > Gmatches;
  //my own feature_match algorithm,used left-right check 
  Gmatches1 = matching( descriptors_1,descriptors_2);
  cout<<"number of point matched: "<<Gmatches1.size()<<endl;
  cout<<" surf matches"<<endl;
  cout<<"Press any button on imshow window to proceed"<<endl<<endl;
  show_matches("surf matches", img_1, keypoints_1, img_2, keypoints_2, Gmatches1 );

   //based on my own feature matching Gmatches1, using the library to find out all outliers
  Gmatches2 = test( img_1, keypoints_1, img_2, keypoints_2, Gmatches1 );
  cout<<"number of inliner point matched: "<<Gmatches2.size()<<endl;
  cout<<"surf inliner matches"<<endl;
  cout<<"Press any button on imshow window to proceed"<<endl<<endl;
  show_matches("surf inliner matches", img_1, keypoints_1, img_2, keypoints_2, Gmatches2 );

  //as the feature point maybe less, just some checking to decide use which Dmatch
  if(Gmatches2.size() < 50)
  {
    Gmatches = Gmatches1;
  }
  else
  {
     Gmatches = Gmatches2;
  }
  //show 15 points for drawEpipolarLines

  cout<<endl<<" ---Task 1.3. Epipolar Geometry---"<<endl;
  int size_goodpoint = 15;
  if( Gmatches.size() < 15 )
  {
    size_goodpoint = Gmatches.size();
  }
  //sort the diestance in Dmatch
  std::sort(Gmatches.begin(), Gmatches.end());
  Best_matches = Gmatches;
  Best_matches.resize(size_goodpoint);

  std::vector<KeyPoint> goodkeypoints_1, goodkeypoints_2;
  for(int d = 0 ; d< Best_matches.size();d++)
  {
    goodkeypoints_1.push_back(cv::KeyPoint( keypoints_1[Best_matches[d].queryIdx].pt, 1.f));
    goodkeypoints_2.push_back(cv::KeyPoint(keypoints_2[Best_matches[d].trainIdx].pt,1.f));
  }

  int ptCount = (int)Best_matches.size();
  Mat p1(ptCount, 2, CV_32F);
  Mat p2(ptCount, 2, CV_32F);
  std::vector< Point2f > po1;
  std::vector< Point2f > po2;


  Point2f pt;
  for (int i=0; i<ptCount; i++)
  {
      pt = keypoints_1[Best_matches[i].queryIdx].pt;
      p1.at<float>(i, 0) = pt.x;
      po1.push_back(Point2f(pt.x,pt.y));
      p1.at<float>(i, 1) = pt.y;

      pt = keypoints_2[Best_matches[i].trainIdx].pt;
      p2.at<float>(i, 0) = pt.x;
      po2.push_back(Point2f(pt.x,pt.y));
      p2.at<float>(i, 1) = pt.y;
  }
  // cout<<p1<<endl<<endl;
  // cout<<p2<<endl<<endl;

  //Using Opencv Library to check value
  // Mat  m_Fundamental = findFundamentalMat(p1, p2, FM_8POINT);
  // cout<<m_Fundamental<<endl;

  Mat m_Fundamental =findFundamental(p1, p2,size_goodpoint);
  cout<<"the fundamental matrix calculated:  "<<endl<<m_Fundamental<<endl;

  Mat m_Fundamental2 =findFundamental(p1.rowRange(0,7), p2.rowRange(0,7),7);
  // Mat R,T;
  // rotation(m_Fundamental,R,T);
  string f2 = "EpipolarLines";
  cout<<"using my own fundamental matrix to draw Epipolar Lines"<<endl<<endl;
  cout<<"Press any button on imshow window to proceed"<<endl<<endl;
 // drawEpipolarLines(f2, m_Fundamental, img_1,img_2, po1, po2, 0);


  cout<<endl<<" ---Task 1.4. Sparse 3D Points (Two-view Triangulation)---"<<endl;
  string file = i1+"vs"+i2+".ply";
  PLYfrom2D(file, img_1, keypoints_1, img_2, keypoints_2, Gmatches1 );

  cout<<"the correspond ply file is generated"<<endl<<endl;
  cout<<"The name is: "<<file<<endl;

}

void fundaemtaleachpairall(string s1, string s2)
{ 
  
  string i1 = s1;
  string i2 = s2;

  Mat img_1 = imread( i1+".png", IMREAD_GRAYSCALE );
  Mat img_2 = imread( i2+".png", IMREAD_GRAYSCALE );


  int minHessian = 1400; 
  Ptr<SURF> detector = SURF::create();
  detector->setHessianThreshold(minHessian);
  detector->setExtended (true);

  std::vector<KeyPoint> surf_keypoints_1, surf_keypoints_2;
  Mat surf_descriptors_1, surf_descriptors_2;

  detector->detectAndCompute( img_1, Mat(), surf_keypoints_1, surf_descriptors_1 );
  detector->detectAndCompute( img_2, Mat(), surf_keypoints_2, surf_descriptors_2 );

  std::vector<KeyPoint> keypoints_1, keypoints_2;
  Mat descriptors_1, descriptors_2;

  keypoints_1 = surf_keypoints_1, keypoints_2 = surf_keypoints_2;
  descriptors_1 = surf_descriptors_1 , descriptors_2 = surf_descriptors_2;

  std::vector< DMatch > Gmatches1;
  std::vector< DMatch > Gmatches2;
  std::vector< DMatch > Best_matches;
  
  std::vector< DMatch > Gmatches;
  Gmatches1 = matching( descriptors_1,descriptors_2);
  cout<<"number of point matched: "<<Gmatches1.size()<<endl;
  
  Gmatches2 = test( img_1, keypoints_1, img_2, keypoints_2, Gmatches1 );
  cout<<"number of inliner point matched: "<<Gmatches2.size()<<endl;
 
  if(Gmatches2.size() < 50)
  {
    Gmatches = Gmatches1;
  }
  else
  {
     Gmatches = Gmatches2;
  }
  
  int size_goodpoint = 15;
  if( Gmatches.size() < 15 )
  {
    size_goodpoint = Gmatches.size();
  }
  std::sort(Gmatches.begin(), Gmatches.end());
  Best_matches = Gmatches;
  Best_matches.resize(size_goodpoint);

  std::vector<KeyPoint> goodkeypoints_1, goodkeypoints_2;
  for(int d = 0 ; d< Best_matches.size();d++)
  {
    goodkeypoints_1.push_back(cv::KeyPoint( keypoints_1[Best_matches[d].queryIdx].pt, 1.f));
    goodkeypoints_2.push_back(cv::KeyPoint(keypoints_2[Best_matches[d].trainIdx].pt,1.f));
  }

  int ptCount = (int)Best_matches.size();
  Mat p1(ptCount, 2, CV_32F);
  Mat p2(ptCount, 2, CV_32F);
  std::vector< Point2f > po1;
  std::vector< Point2f > po2;

  Point2f pt;
  for (int i=0; i<ptCount; i++)
  {
      pt = keypoints_1[Best_matches[i].queryIdx].pt;
      p1.at<float>(i, 0) = pt.x;
      po1.push_back(Point2f(pt.x,pt.y));
      p1.at<float>(i, 1) = pt.y;

      pt = keypoints_2[Best_matches[i].trainIdx].pt;
      p2.at<float>(i, 0) = pt.x;
      po2.push_back(Point2f(pt.x,pt.y));
      p2.at<float>(i, 1) = pt.y;
  }

  Mat m_Fundamental =findFundamental(p1, p2,size_goodpoint);
  cout<<"the fundamental matrix calculated:  "<<endl<<m_Fundamental<<endl;

  string file = i1+"vs"+i2+".ply";
  PLYfrom2D(file, img_1, keypoints_1, img_2, keypoints_2, Gmatches1 );

  cout<<"the correspond ply file is generated"<<endl<<endl;
  cout<<"The name is: "<<file<<endl;

}

int main( int argc, char** argv )
{ 
  cout<<"Comp5421 course Project 1: 3D rescoustruction"<<endl;
  cout<<"Lai Chi Kin Benjamin std:20118346"<<endl;
  cout<<endl<<"Here is image list(all are in PNG format):"<<endl<<endl;
  cout<<"0000 | 0001 | 0002 | 0003 | 0004 | 0005 | 0006 | 0007 | 0008 | 0009 | 0010"<<endl<<endl;
  cout<< "please choose two consecutive image pair"<<endl;
  cout<<"example: 0000 0001   ...  or 0004 0005"<<endl;

    cout<<endl<<"another input type is: 'all''" <<endl;
cout<<"when input 'all' for first input, it will proceed to task 2 :)"<<endl;
  string s1,s2;
  cout<<"input the first image:"<<endl;
  getline(cin, s1);
  cout<<"input the second image:"<<endl;
  getline(cin, s2);
  cout<<"you have input "<<s1<<" and "<<s2<<endl;

  if((s1 == "all"))
  {   
     cout<<"now doing "<<endl;
      string input[11] = {"0000",
                          "0001",
                          "0002",
                          "0003",
                          "0004",
                          "0005",
                          "0006",
                          "0007",
                          "0008",
                          "0009","0010"};
      for(int i = 0; i<10; i++)
      {
            cout<<"now doing "<<input[i]<<" "<<input[i+1]<<" image pair"<<endl;
            fundaemtaleachpairall(input[i], input[i+1]);
      }
  }
  else{
      Mat img_1 = imread( s1+".png", IMREAD_GRAYSCALE );
      Mat img_2 = imread( s2+".png", IMREAD_GRAYSCALE );
      if( !img_1.data || !img_2.data )
      {
         cout<<"opps, cannot find these images wor, check the input again?!"<<endl;
         return -1; 
      }

      fundaemtaleachpair(s1, s2);

  }



  return 0;
}

void show_matches(String title, Mat img_1, std::vector<KeyPoint> keypoints_1, Mat img_2,std::vector<KeyPoint> keypoints_2,  std::vector< DMatch > Gmatches)
{

  Mat img_matches;
  drawMatches( img_1, keypoints_1, img_2, keypoints_2, Gmatches, img_matches );
  //-- Show detected matches
  namedWindow(title, 0);
  imshow(title, img_matches );
  resizeWindow(title, 1000,1000);
  waitKey(0);
  destroyWindow(title);

}

void show_keypoints(String title, Mat img_1, std::vector<KeyPoint> keypoints_1, Mat img_2,std::vector<KeyPoint> keypoints_2)
{
  string title1 = title;
  string title2 = title;
  Mat img_keypoints_1; Mat img_keypoints_2;
  drawKeypoints( img_1, keypoints_1, img_keypoints_1, Scalar::all(-1), DrawMatchesFlags::DEFAULT );
  drawKeypoints( img_2, keypoints_2, img_keypoints_2, Scalar::all(-1), DrawMatchesFlags::DEFAULT );

  title1.append("Keypoints 1");
  title2.append("Keypoints 2");

  namedWindow(title1, 0);
  namedWindow(title2, 0);
  imshow(title1, img_keypoints_1 );
  resizeWindow(title1, 1000,1000);
  imshow(title2, img_keypoints_2 );
  resizeWindow(title2, 1000,1000);
  waitKey(0);
  destroyWindow(title1);
  destroyWindow(title2);
}



std::vector< DMatch > test( Mat img_1,
                            vector<KeyPoint> m_LeftKey,
                            Mat img_2,
                            vector<KeyPoint> m_RightKey,
                            vector<DMatch> m_Matches
)
{
  std::vector< DMatch > good_matches;
  int ptCount = (int)m_Matches.size();
  Mat p1(ptCount, 2, CV_32F);
  Mat p2(ptCount, 2, CV_32F);

  Point2f pt;
  for (int i=0; i<ptCount; i++)
  {
      pt = m_LeftKey[m_Matches[i].queryIdx].pt;
      p1.at<float>(i, 0) = pt.x;
      p1.at<float>(i, 1) = pt.y;
    
      pt = m_RightKey[m_Matches[i].trainIdx].pt;
      p2.at<float>(i, 0) = pt.x;
      p2.at<float>(i, 1) = pt.y;
  }

  Mat m_Fundamental;
  vector<uchar> m_RANSACStatus;
  m_Fundamental = findFundamentalMat(p1, p2, m_RANSACStatus, FM_RANSAC);

  for (int i=0; i<ptCount; i++)
  {
      if (m_RANSACStatus[i] == 1)
      {
            good_matches.push_back(m_Matches[i]);
      }

  }

  return good_matches;

}


void PLYfrom2D( String file,
                Mat img_1,
                vector<KeyPoint> m_LeftKey,
                Mat img_2,
                vector<KeyPoint> m_RightKey,
                vector<DMatch> m_Matches
)
{
  std::vector< DMatch > good_matches;
  int ptCount = (int)m_Matches.size();
  Mat p1(ptCount, 2, CV_32F);
  Mat p2(ptCount, 2, CV_32F);

  Point2f pt;
  for (int i=0; i<ptCount; i++)
  {
      pt = m_LeftKey[m_Matches[i].queryIdx].pt;
      p1.at<float>(i, 0) = pt.x;
      p1.at<float>(i, 1) = pt.y;
    
      pt = m_RightKey[m_Matches[i].trainIdx].pt;
      p2.at<float>(i, 0) = pt.x;
      p2.at<float>(i, 1) = pt.y;
  }

  Mat m_Fundamental;
  vector<uchar> m_RANSACStatus;
  m_Fundamental = findFundamentalMat(p1, p2, m_RANSACStatus, FM_RANSAC);

  int OutlinerCount = 0;
  for (int i=0; i<ptCount; i++)
  {
      if (m_RANSACStatus[i] == 1)
      {
            good_matches.push_back(m_Matches[i]);
      }
      else
      {
         OutlinerCount++;
      }
  }

  vector<Point2f> m_LeftInlier;
  vector<Point2f> m_RightInlier;
  vector<DMatch> m_InlierMatches;

  int InlinerCount = ptCount - OutlinerCount;
  m_InlierMatches.resize(InlinerCount);
  m_LeftInlier.resize(InlinerCount);
  m_RightInlier.resize(InlinerCount);
  InlinerCount = 0;
  for (int i=0; i<ptCount; i++)
  {
    if (m_RANSACStatus[i] != 0)
    {
        m_LeftInlier[InlinerCount].x = p1.at<float>(i, 0);
        m_LeftInlier[InlinerCount].y = p1.at<float>(i, 1);
        m_RightInlier[InlinerCount].x = p2.at<float>(i, 0);
        m_RightInlier[InlinerCount].y = p2.at<float>(i, 1);
        m_InlierMatches[InlinerCount].queryIdx = InlinerCount;
        m_InlierMatches[InlinerCount].trainIdx = InlinerCount;
        InlinerCount++;
    }
  }

  // vector<KeyPoint> key1(InlinerCount);
  // vector<KeyPoint> key2(InlinerCount);
  // KeyPoint::convert(m_LeftInlier, key1);
  // KeyPoint::convert(m_RightInlier, key2);

  Mat R,T;
  rotation(m_Fundamental,R,T,0);

  cv::hconcat(R, T, R);

  Mat P1 = Mat::eye(3, 4, CV_64F);

//  Mat point1matrix(m_LeftInlier);

  //cout<<point1matrix<<endl;
  //Mat point2matrix(m_RightInlier);

  // cout<<"test"<<endl;
  Mat pnts4D;

  Mat P = cameraMatrix() * P1;

  cout<<endl<<"The projection matrix for camera 1"<<endl<<P<<endl;
  Mat P2 = cameraMatrix()*R;
  cout<<endl<<"The projection matrix for camera 2"<<endl<<P2<<endl;

  triangulatePoints(P, P2,m_LeftInlier, m_RightInlier, pnts4D );

  // cout<<pnts4D.t()<<endl;

  Mat dd = pnts4D.t();
  for(int i = 0; i < dd.rows; i++)
  {
      if(dd.at<float>(i,3) < 0.0)
      {   
          cout<<"find new"<<endl;
          rotation(m_Fundamental,R,T,1);
          cv::hconcat(R, T, R);
          Mat P2 = cameraMatrix()*R;
          triangulatePoints(P, P2,m_LeftInlier, m_RightInlier, pnts4D );
          break;
      }

  }
  // cout<<pnts4D.t()<<endl;
  // cout<<dd.at<float>(0,0)<<endl;

  // cout<<dd.at<float>(0,3)<<endl;
  Mat pnts3D;
  convertPointsFromHomogeneous(pnts4D.t() , pnts3D);
  // cout<<pnts3D<<endl;
  viz::writeCloud(file,pnts3D);


}

void rotation(Mat F, Mat &R, Mat &T, int r1)
{
  Mat A = cameraMatrix();
    cout<<endl<<"The camera matrix:"<<endl<<A<<endl;
  Mat E = A.t() * F * A;

  cout<<endl<<"The essential matrix:"<<endl<<E<<endl;

  SVD decomp = SVD(E,4);

  Mat U = decomp.u;

  Mat V = decomp.vt; 

  Mat W(3, 3, CV_64F, Scalar(0));
  W.at<double>(0, 1) = -1;
  W.at<double>(1, 0) = 1;
  W.at<double>(2, 2) = 1;

// cout<<decomp.w<<endl;;
  Mat R1 =  U * W * V; 
  Mat R2 =  U * W.t() * V; 


  Mat T1 = U.col(2); 
  Mat T2 = -1* T1;

  // cout<<R1<<endl;
  // cout<<R2<<endl;

  // cout<<T1<<endl;

  // cout<<T2<<endl;
  if(r1 == 0)
  {
      R = R1;
  }
  else{
      R = R2;
  }
  // R = R1;

  cout<<endl<<"The rotational matrix"<<endl<<R<<endl;
  T = T2;
  cout<<endl<<"The translation vector"<<endl<<T<<endl;
  Mat test = Mat(4,1, CV_64F, Scalar(100));
  // Mat P1 = R1;
  // cv::hconcat(P1, -R1*T1, P1);

  // Mat P2 = R2;
  // cv::hconcat(P2, -R2*T1, P2);

  // Mat P3 = R1;
  // cv::hconcat(P3, -R1*T2, P3);

  // Mat P4 = R2;
  // cv::hconcat(P4, -R2*T2, P4);

// cout<<P1<<endl<<P2<<endl<<P3<<endl<<P4<<endl;


// cout<<P1*test<<endl<<P2*test<<endl<<P3*test<<endl<<P4*test<<endl;
}

Mat cameraMatrix()
{
  Mat cm =   (Mat_<double>(3,3) <<  2759.48, 0,1520.69,0, 2764.16, 1006.81, 0, 0,1);
  return cm;


}