# comp5421_project1
Project 1 - 3D Reconstruction:
In this project, you are required to recover a 3D scene from multiple un-calibrated images (dataset link here). The project contains two required tasks and one bonus task. You can use libraries e.g. OpenCV and Eigen to facilitate your implementation.
 
Task 1: Two-view geometry (75%)
Given one image pair (e.g. 0005.jpg and 0006.jpg), your first task is to compute the epipolar geometry for this image pair, during which you are required to implement the following:
1.    Feature Detection. Detect image features using SIFT, SURF, corners or any learned feature detectors (try at least 2 types). Compute the corresponding descriptors and visualize the features. (20%)
 
2.    Feature Matching. Compute the nearest neighbor using Euclidean Distance and find the matching correspondence (Using either enumeration or approximation method). Visualize the corresponding pairs. (10%)
Note: 1. You should implement the matching algorithm yourself.
          2. Using left-right check could significantly filter out the mismatches, i.e., for the feature p in the left image and its best matching feature q in the right image, p should also be the best matching feature of q.
 
3.    Epipolar Geometry. Compute the fundamental matrix using: a) > 8 pairs of points, b) 7 pairs of points. Visualize the epipolar geometry by drawing epipolar lines of the 15 best feature matches. (25%)
Note: 1. You should implement the algorithm yourself.
 
4.    Sparse 3D Points (Two-view Triangulation). Compute the 3D points for each feature match that satisfy the epipolar geometry. Visualize the points by saving into a .ply file (use meshlab to open the .ply file). (10%)
Note: 1. You may presume the intrinsic parameters are known as:
                        2759.48           0                      1520.69
                        0                      2764.16           1006.81
                        0                      0                      1
          2. RANSAC could be used for the robust estimation of the fundamental matrix.
          3. Triangulation 3D points with the camera matrix.
 
Task 2: Structure from Motion (25%)
1.    Compute fundamental matrices for all images pairs.
2.    Compute a global consistent camera system for all images.
Notes:
1.     One solution is through incremental reconstruction. You can set the first camera matrix to be K [I, 0], and then compute all the relative rotation R__ij and translation tij between each camera pair (see below). Given the extrinsic [R_i, ti] of camera i and the relative transformation R__ij and tij between camera i and j, the extrinsic of camera j can be expressed as:
Rj = Rij Ri
tj = Rij ti + tij
2.     About computing R__ij and tij: in task 1.4 you have computed the relative rotation and translation. However, the translation decomposed from the essential matrix is always normalized that || tij || = 1 (without scale information). So we must recover the scale sij of each relative translation before chaining all cameras together. The true tij should be: tij := sij * tij
An easy method is depth checking. Suppose you have processed two image pairs (I0, I1) and (I1, I2) and have established two individual camera systems S01 (with R0 01 t0 01 R1 01 t1 01) and S12 (with R1 12 t1 12 R2 12 t2 12), you need to find two matches (f0, f1), (f1, f2) in these two image pairs that refer to the same 3D point. We denote the 3D points of these two matches in S01, S12 be P01 and P12, their depths in image I1 should be the same:
                                    d1 01 = (R1 01 P01 + t1 01).z = d1 12 = (R1 12 P12 + t1 12).z
Which could be used to derive the relationship of s01 and s12 .
 
There are also other complicated methods, such as solving a tri-focal tensor (see Multiview Geometry in Computer Vision) or utilizing five-point algorithm with three views (see An Efficient Solution to the Five-Point Relative Pose Problem). In addition, some open-sourced libraries such as OpenMVG and colmap have already implemented this algorithm, which you can use as a reference.
