#include <stdio.h>
#include <iostream>
#include "opencv2/core.hpp"

using namespace cv;

std::vector< DMatch > matching(  Mat descriptors_1, Mat descriptors_2)
{
    std::vector< DMatch > good_matches;

    double** min_distance = (double **) malloc(descriptors_1.rows * sizeof(double *));
    for(int w = 0; w < descriptors_1.rows; w++)
    {
        min_distance[w] = (double *) malloc(2* sizeof(double));

        for(int i = 0; i < descriptors_2.rows; i++)
        {     
            double norm_dist =  cv::norm(descriptors_1.row(w)-descriptors_2.row(i));
            if(i == 0)
            {
                min_distance[w][0] = norm_dist;
                min_distance[w][1] = i;
            }
            else if( min_distance[w][0] > norm_dist)
            {
                min_distance[w][0] = norm_dist;
                min_distance[w][1] = i;
            }
        }
        //check is that the optimal solution
        double check_min[2];
        for(int i = 0; i < descriptors_1.rows; i++)
        {     
            double norm_dist =  cv::norm(descriptors_1.row(i)-descriptors_2.row(min_distance[w][1]));

            if(i == 0)
            {
            check_min[0] = norm_dist;
            check_min[1] = i;
            }
            else if( check_min[0] > norm_dist)
            {
            check_min[0] = norm_dist;
            check_min[1] = i;
            }
        }
        if(check_min[1] != w )
        {     
                min_distance[w][0] = 0;
                min_distance[w][1] = -1;
        }
        else{
            if(min_distance[w][0] < 0.35)
            {
                good_matches.push_back( DMatch(w,  min_distance[w][1] , min_distance[w][0]));
            }
        }
    }
    free(min_distance);
    return good_matches;
}