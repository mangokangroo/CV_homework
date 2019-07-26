# Pseudo RANSAC Code
'''
Input:
2 sets of points, group1 and group 2,filtered by KNN method and
refined by a threshold on distance

Procedure:
final_homography = []
all_inliers = []
inlier_not_change = 0
max_inlier_num = 0
not_end = True
max_inter_num = 2000

while not_end & iter < max_inter_num:
    pick_inliers = randomly select 4 pairs of points   # as assumed inliers
    homography_inliers = calculate the homography of the selected 4 pairs of points
    group1_rest_homo = transform the rest of points in group1 by homography_inliers
    new_inliers = []
    for i in len(group_rest_homo):
        if group1_rest_homo(i) - group2_rest(i) <= threshold:
            new_inliers.append(group1_rest_homo(i)
        else:
            pass

    all_inliers = pick_inliers.append(new_inliers)
    num_all_inliers = len(all_inliers)
    if num_all_inliers > max_inlier_num:
        max_inlier_num = num_all_inliers
        final_homography = homography_inliers
        inlier_not_change = 0
    else:
        inlier_not_change +=1

    iter += 1
    if inlier_not_change > 1000:
        not_end = False


Output:
final_homography, all_inliers
'''