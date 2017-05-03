#include <stdio.h>
#include <math.h>
#include <iostream>
#include "opencv2/core.hpp"

using namespace std;
using namespace cv;

Mat findFundamental(Mat k1, Mat k2, int N)
{ 
  Mat F =  Mat(3,3, CV_64F, double(0));
  //8 point algorithm
  if(N > 7)
  {
    cout<<"8 point algorithm"<<endl<<endl;
    Mat k1x = k1.col(0);
    Mat k1y = k1.col(1);
    Mat k2x = k2.col(0);
    Mat k2y = k2.col(1);

    Mat A  = Mat(N,9, CV_64F, double(1));

    Mat A1 = k1x.mul(k2x).t();
    Mat A2 = k1x.mul(k2y).t();
    Mat A3 = k1x.mul(1).t();
    Mat A4 = k1y.mul(k2x).t();
    Mat A5 = k1y.mul(k2y).t();
    Mat A6 = k1y.mul(1).t();
    Mat A7 = k2x.mul(1).t();
    Mat A8 = k2y.mul(1).t();

    Mat B;
    B.push_back(A1);
    B.push_back(A2);
    B.push_back(A3);
    B.push_back(A4);
    B.push_back(A5);
    B.push_back(A6);
    B.push_back(A7);
    B.push_back(A8);
    B.push_back(A1);

    A = B;

    A.row(8).setTo(1);

    Mat S,U,Vt;
    SVD::compute(A.t(),S,U,Vt);

    for(int i = 0; i< 3; i++)
    {
      F.at<double>(i,0)= Vt.at<float>(8,i*3);
      F.at<double>(i,1)= Vt.at<float>(8,i*3+1);
      F.at<double>(i,2)= Vt.at<float>(8,i*3+2);
        
    }
    SVD::compute(F,S,U,Vt);

    Mat s1 = Mat::eye(3,3, CV_64F);

    s1.at<double>(0,0) = S.at<double>(0,0);
    s1.at<double>(1,1) = S.at<double>(1,0);
    s1.at<double>(2,2) = 0.0;
    F = U * s1 * Vt;
    F = F.t();

  }
  else if(N == 7)//7 point algorithm
  { 
    cout<<endl<<"7 point algorithm"<<endl;
    Mat k1x = k1.col(0);
    Mat k1y = k1.col(1);
    Mat k2x = k2.col(0);
    Mat k2y = k2.col(1);

    Mat A  = Mat(N,9, CV_64F, double(1));

    Mat A1 = k1x.mul(k2x).t();
    Mat A2 = k1x.mul(k2y).t();
    Mat A3 = k1x.mul(1).t();
    Mat A4 = k1y.mul(k2x).t();
    Mat A5 = k1y.mul(k2y).t();
    Mat A6 = k1y.mul(1).t();
    Mat A7 = k2x.mul(1).t();
    Mat A8 = k2y.mul(1).t();

    Mat B;
    B.push_back(A1);
    B.push_back(A2);
    B.push_back(A3);
    B.push_back(A4);
    B.push_back(A5);
    B.push_back(A6);
    B.push_back(A7);
    B.push_back(A8);
    B.push_back(A1);

    A = B;

    A.row(8).setTo(1);

    Mat S,U,Vt;
    SVD::compute(A.t(),S,U,Vt,4);

    Mat FF1 = Mat(3,3, CV_64F, double(0));
    Mat FF2 =  Mat(3,3, CV_64F, double(0));
    // cout<<A.t()<<endl;
    // cout<<Vt.t()<<endl;
    // cout<<U<<endl;
    // cout<<S<<endl;
    // Mat I = Mat::eye(7, 9, CV_32F);

    // I = Mat::diag(S);

    // cv::hconcat(I, S, I);

    // cv::hconcat(I, S, I);
    // I.col(7).setTo(0);
    // I.col(8).setTo(0);
    // cout<<I<<endl;
    // cout<<U*I*Vt<<endl;
    for(int i = 0; i< 3; i++)
    {
      FF1.at<double>(i,0)= Vt.at<float>(8,i*3);
      FF1.at<double>(i,1)= Vt.at<float>(8,i*3+1);
      FF1.at<double>(i,2)= Vt.at<float>(8,i*3+2);

      FF2.at<double>(i,0)= Vt.at<float>(7,i*3);
      FF2.at<double>(i,1)= Vt.at<float>(7,i*3+1);
      FF2.at<double>(i,2)= Vt.at<float>(7,i*3+2);
        
    }

    // cout<<FF1<<endl<<FF2<<endl;
    Mat FFF[2];
    // Mat d1 =  (Mat_<double>(3,3) <<  1.46749081715488e-05,	0.000365983832855093,	-0.464337540146174,
    // -0.000362241038918125,	1.28660247701681e-05,	0.549530787668022,
    // 0.428595262436388,	-0.546545768420997,	-0.00034743420405599);

    // Mat d2 =  (Mat_<double>(3,3) << 1.61841055556898e-07,	1.36187903368991e-06,	-0.00118219242674904,
    // -1.36724196074352e-06,	1.41701172068860e-07,	-0.000397426667282590,
    // 0.000496204906946844,	0.000358206662012326,	0.999999034968543);
    FFF[0] = FF1;
    FFF[1] = FF2;
    // FFF[0] = d1;
    // FFF[1] = d2;
    // cout<<FFF[0]<<endl;
    double D[2][2][2];

    for(int x = 0; x<2; x++)
    {
      for(int y = 0; y<2; y++)
      {
        for(int  z= 0; z<2; z++)
        {
          Mat Z = FFF[x].col(0);

          cv::hconcat(Z, FFF[y].col(1),Z);
          cv::hconcat(Z, FFF[z].col(2),Z);

          D[x][y][z] = determinant(Z);

          // cout<<D[x][y][z]<<endl;

        }
      }
    } 
    

  // double a[4] = {-D[1][0][0]+D[0][1][1]+D[0][0][0]+D[1][1][0]+D[1][0][1]-D[0][1][0]-D[0][0][1]-D[1][1][1],
  //         D[0][0][1]-2*D[0][1][1]-2*D[1][0][1]+D[1][0][0]-2*D[1][1][0]+D[0][1][0]+3*D[1][1][1],
  //         D[1][1][0]+D[0][1][1]+D[1][0][1]-3*D[1][1][1],
  //         D[1][1][1]};

  double a[4] =  { D[1][1][1],
  D[1][1][0]+D[0][1][1]+D[1][0][1]-3*D[1][1][1],
  D[0][0][1]-2*D[0][1][1]-2*D[1][0][1]+D[1][0][0]-2*D[1][1][0]+D[0][1][0]+3*D[1][1][1],
  -D[1][0][0]+D[0][1][1]+D[0][0][0]+D[1][1][0]+D[1][0][1]-D[0][1][0]-D[0][0][1]-D[1][1][1]};
  
  Mat input = Mat(4,1,CV_64F,&a);

  cout<<"coefficients:"<<endl<<input<<endl;
  Mat result;
  solvePoly(input,result);

  cout<<"roots"<<endl<<result<<endl;

  Mat F1 = result.at<double>(2,0) * FF1 + (1 - result.at<double>(2,0)) * FF2;
  Mat F2 = result.at<double>(2,0) * FF1 + (1 - result.at<double>(2,0)) * FF2;
  Mat F3 = result.at<double>(2,0) * FF1 + (1 - result.at<double>(2,0)) * FF2;

  // cout<<result.at<double>(2,0)<<endl;
  cout<<endl<<"result of 7-point algorthm"<<endl<<F1<<endl<<F2<<endl<<F3<<endl;
  F = F1;

  }
  return F;

}