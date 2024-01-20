import cv2
import numpy as np

def detect_and_display_circles(image_path, minRadius, maxRadius, minDist,
                               circularity_threshold=0.4,
                               max_sticker_number=None,
                               trim_border=None,
                               ):
    """
    This function detects the circles in the input image and displays the detected circles on the raw image.
    :param image_path: path to the input image, e.g. "Images\rgb2.png"
    :param minRadius: minimum radius of the circles to be detected
    :param maxRadius: maximum radius of the circles to be detected
    :param minDist: minimum ct-to-ct distance
    :param circularity_threshold: a threshold for circularity, can adjust based on requirement
    :param max_sticker_number: if specified, the function will iteratively adjust the param2 of Hough Circle Transform
    :param trim_border: if specified, the function will trim the border of the image by the given percentage
    :return: None"""

    ### Read the input image and convert to grayscale
    # Read the image and convert to grayscale
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image_current = gray.copy()
    cv2.imshow("Initial Grayscale Image", gray)
    cv2.waitKey(0)

    ### Preprocessing - use Canny edge detection and impainting to find filled patches within a certain area
    # Apply Canny Edge Detection
    edges = cv2.Canny(image_current, 100, 200)
    cv2.imshow("Canny Edges", edges)
    cv2.waitKey(0)

    # Find contours
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Filter contours based on area and circularity
    min_area = np.pi * minRadius ** 2  # Minimum area (considering minimum radius is 10)
    max_area = np.pi * maxRadius ** 2  # Maximum area (considering maximum radius is 30)
    filtered_contours = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if min_area <= area <= max_area:
            perimeter = cv2.arcLength(cnt, True)
            circularity = 4 * np.pi * area / (perimeter ** 2) if perimeter > 0 else 0
            if circularity > circularity_threshold:  # A threshold for circularity, can adjust based on requirement
                filtered_contours.append(cnt)

    # Create a mask to keep only meaningful patches in the image
    mask = np.zeros_like(gray)
    cv2.drawContours(mask, filtered_contours, -1, (255), thickness=-1)
    cv2.imshow("Filtered Patches", mask)
    cv2.waitKey(0)
    image_current = cv2.bitwise_and(image_current, mask)

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
    param1 = 30  # the higher threshold for the Canny edge detector
    param2 = 10  # The accumulator threshold for the circle centers during detection (small value -> false circles)

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
    # path_to_image = r"Images\gray.png"
    path_to_image = r"Images\gray2.png"
    # path_to_image = r"Images\rgb.png"
    # path_to_image = r"Images\rgb2.png"

    # set mandatory parameters
    minRadius = 5  # minimum radius of the circles to be detected
    maxRadius = 30  # maximum radius of the circles to be detected
    minDist = 20  # minimum ct-to-ct distance
    circularity_threshold = 0.2  # a threshold for circularity

    # set optional parameters (set to None if not needed)
    max_sticker_number = None  # only detect up to 5 stickers
    trim_border = 0.2  # trim 10% of the border (e.g. 0.1 to trim 10% boarders)

    # run the detection
    detect_and_display_circles(path_to_image,
                               minRadius=minRadius,
                               maxRadius=maxRadius,
                               minDist=minDist,
                               circularity_threshold=circularity_threshold,
                               max_sticker_number=max_sticker_number,
                               trim_border=trim_border,
                               )
