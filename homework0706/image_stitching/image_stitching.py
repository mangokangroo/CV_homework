import cv2
import numpy as np


def get_good_matches(feature_points1, feature_points2):
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(feature_points1, feature_points2, k=2)

    # Filter good matches based on a distance of 0.75 between
    # feature points in pairs in 2 images
    feature_points_list = []
    feature_points = []

    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            feature_points_list.append([m])
            feature_points.append(m)
    return feature_points_list, feature_points

def get_inliers(mask_homo, num=10):
    matches_mask = mask_homo.ravel().tolist()    # ravel to 1-D list
    indices = []
    for ind in range(len(matches_mask)):
        if matches_mask[ind] == 1:
            indices.append(ind)
    matches_mask = [0] * len(matches_mask)
    np.random.shuffle(indices)
    indices = indices[:num]   # random sort
    for ind in indices:
        matches_mask[ind] = 1
    return matches_mask


def get_bound(image1_stiching):
    x_min = min(image1_stiching[0][0], image1_stiching[1][0])
    y_min = min(image1_stiching[0][1], image1_stiching[3][1])
    x_max = max(image1_stiching[2][0], image1_stiching[3][0])
    y_max = max(image1_stiching[1][1], image1_stiching[2][1])
    return x_min, y_min, x_max, y_max




# Reading images to stitch
image1 = cv2.imread('Images/mountain1.jpg')
image2 = cv2.imread('Images/mountain2.jpg')

# Find SIFT feature points and corresponding descriptors
sift = cv2.xfeatures2d.SIFT_create()
features_image1, descriptors_image1 = sift.detectAndCompute(image1, None)
features_image2, descriptors_image2 = sift.detectAndCompute(image2, None)

# SIFT feature points display
image1_SIFT_points = cv2.drawKeypoints(image1, features_image1, None)
image2_SIFT_points = cv2.drawKeypoints(image2, features_image2, None)
cv2.imwrite('Results/image1_siftpoints.jpg', image1_SIFT_points)
cv2.imwrite('Results/image2_siftpoints.jpg', image2_SIFT_points)

# Using brute force matching method -- KNN algorithm, to get K nearest
# feature points according to respective descriptors
good_features_list, good_features = get_good_matches(descriptors_image1, descriptors_image2)
image_knn_features = cv2.drawMatchesKnn(image1, features_image1, image2, features_image2, good_features_list, None, flags=2 )
cv2.imwrite('Results/knn_good_matches.jpg', image_knn_features)

# Convert feature points list into array
coord_feature1 = np.array([features_image1[m.queryIdx].pt for m in good_features]).reshape(-1, 1, 2)
coord_feature2 = np.array([features_image2[m.trainIdx].pt for m in good_features]).reshape(-1, 1, 2)

# Get homography after applying RANSAC on well-matched feature points
H, mask = cv2.findHomography(coord_feature1, coord_feature2, cv2.RANSAC)
print('Homography for transforming image1 to image 2 is: ')
print(H)

# Get 10 inliers after RANSAC for display
matchesMask = get_inliers(mask, 10)
inlier_image = cv2.drawMatches(image1, features_image1, image2, features_image2,
                               good_features, None, matchesMask=matchesMask, flags=2)
cv2.imwrite('Result/inlier_matches.jpg', inlier_image)

# Get four corner coordinates of image1 in the plane of stitching
height, width, dim = image1.shape
corners_image1 = np.float32([[0, 0], [0, height-1],
                           [width-1, height-1], [width-1, 0]]).reshape(-1, 1, 2)
corners_image1_mergePlane = np.squeeze(cv2.perspectiveTransform(corners_image1, H)) # remove dim = 1

# Find the max dimensions that image1 can take in the stitching plane
xmin, ymin, xmax, ymax = get_bound(corners_image1_mergePlane)

# To find the shape of output image through the dimension of image1
# on stitching plane and the size of  image1 and image2
stitch_plane_1 = (xmax - xmin, ymax - ymin)
stitch_plane_2 = (len(image2[0])-int(xmin), len(image2)-int(ymin))
final_image_shape = max(stitch_plane_1, stitch_plane_2)

# Get the matrix for translation of image1 from perspective warp
if xmin < 0 and ymin < 0:
    translate = np.float32([[1, 0, -xmin], [0, 1, -ymin], [0, 0, 1]])
elif xmin < 0:
    translate = np.float32([[1, 0, -xmin], [0, 1, 0], [0, 0, 1]])
elif xmin < 0:
    translate = np.float32([[1, 0, 0], [0, 1, -ymin], [0, 0, 1]])
else:
    translate = np.float32([[1, 0, 0], [0, 1, 0], [0, 0, 1]])

# Applying homography to image 1 to warp it to image2 with translation
final_image = cv2.warpPerspective(image1, np.matmul(translate, H), final_image_shape)

# Slicing the image with warped image 1 to place image2 as well
final_image[-int(ymin):-int(ymin)+len(image2), -int(xmin):-int(xmin)+len(image2[0])] = image2
cv2.imwrite('Results/panorama.jpg', final_image)