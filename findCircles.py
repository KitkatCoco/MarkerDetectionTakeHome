import cv2
import numpy as np

def detect_and_display_circles(image_path,
                               max_sticker_number=None,
                               trim_border=None,
                               ):
    """
    This function detects the circles in the input image and displays the detected circles on the raw image.
    :param image_path: path to the input image, e.g. "Images\rgb2.png"
    :param max_sticker_number: if specified, the function will iteratively adjust the param2 of Hough Circle Transform
    :param trim_border: if specified, the function will trim the border of the image by the given percentage
    :param truncate_darkest_pixels: if specified, the function will truncate the darkest pixels by the given percentile
    """

    ### Read the input image and convert to grayscale
    # Read the image and convert to grayscale
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image_current = gray.copy()
    cv2.imshow("Initial Grayscale Image", gray)
    cv2.waitKey(0)


    # ### Preprocessing - truncate the darkest pixels using given percentage (e.g. 10%)
    # if truncate_darkest_pixels is not None:
    #
    #     # get the histogram of the gray scale image
    #     hist = cv2.calcHist([image_current], [0], None, [256], [0, 256])
    #     hist = hist.flatten()
    #     hist.sort()
    #
    #     # get the top cutoff percentile value
    #     top_cutoff_value = np.percentile(hist, truncate_darkest_pixels*100)
    #     adaptive_thresh_value = np.where(hist >= top_cutoff_value)[0][0]
    #
    #     # apply the thresholding and visualize
    #     _, gray_truncated = cv2.threshold(image_current, adaptive_thresh_value, 255, cv2.THRESH_TOZERO)
    #     image_current = gray_truncated.copy()
    #     cv2.imshow("Grayscale Image After Adaptive Truncating", image_current)
    #     cv2.waitKey(0)

    ### Preprocessing - trim and zero-pad the border, assuming ROI should be centered
    if trim_border is not None:

        # Extract the image dimensions and define the padding zone
        height, width = image_current.shape
        pad_x = int(width * trim_border)
        pad_y = int(height * trim_border)
        gray_truncated_center = image_current.copy()

        # Set the outer border region to zero
        gray_truncated_center[:pad_y, :] = 0                     # Top border
        gray_truncated_center[-pad_y:, :] = 0                    # Bottom border
        gray_truncated_center[:, :pad_x] = 0                     # Left border
        gray_truncated_center[:, -pad_x:] = 0

        # Visualize the padded image
        image_current = gray_truncated_center.copy()
        cv2.imshow("Grayscale Image After Border Zeroing", image_current)
        cv2.waitKey(0)

    ### Apply Hough Circle Transform
    # set up initial parameters for Hough Circle Transform
    minRadius = 10  # lower bound of the circle's radius
    maxRadius = 30  # upper bound of the circle's radius
    minDist = minRadius * 2  # minimum ct-to-ct distance, set to 2x the min radius
    param1 = 30  # the higher threshold for the Canny edge detector
    param2 = 20  # The accumulator threshold for the circle centers during detection (small value -> false circles)

    # Initial run of Hough Circle Transform
    circles = cv2.HoughCircles(image_current, cv2.HOUGH_GRADIENT, 1,
                               minDist=minDist,  # minimum ct-to-ct distance, set to 3x the max radius
                               minRadius=minRadius,
                               maxRadius=maxRadius,
                               param1=param1,
                               param2=param2,
                               )

    # if a max sticker number is specified, then iteratively adjust the param2 to get the desired number of stickers
    if max_sticker_number is not None:
        param2_adaptive = param2
        while len(circles[0]) > max_sticker_number and param2_adaptive < 100:
            if len(circles[0]) > max_sticker_number:
                param2_adaptive += 1
            else:
                param2_adaptive -= 1
            circles = cv2.HoughCircles(image_current, cv2.HOUGH_GRADIENT, 1,
                                       minDist=minDist,  # minimum ct-to-ct distance, set to 3x the max radius
                                       minRadius=minRadius,
                                       maxRadius=maxRadius,
                                       param1=param1,
                                       param2=param2_adaptive,
                                       )

    ### Visualization Results
    if circles is not None:
        print(f"Found {len(circles[0])} circles.")
        circles = np.round(circles[0, :]).astype("int")
        for (x, y, r) in circles:
            cv2.circle(image, (x, y), r, (0, 255, 0), 4)
            cv2.circle(image, (x, y), 1, (0, 100, 255), 3)
        cv2.imshow("Detected Circles", image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print("No circles detected.")




if __name__ == "__main__":

    # specify the path to the image
    path_to_image = r"Images\rgb2.png"

    # set optional parameters
    max_sticker_number = 5  # only detect up to 5 stickers
    trim_border = 0.2  # trim 10% of the border (e.g. 0.1 to trim 10% boarders)
    truncate_darkest_pixels = None  # truncate the darkest pixels (e.g. 0.2 to truncate the darkest 20% pixels)

    # run the detection
    detect_and_display_circles(path_to_image,
                               max_sticker_number=max_sticker_number,
                               trim_border=trim_border,
                               truncate_darkest_pixels=truncate_darkest_pixels,
                               )
