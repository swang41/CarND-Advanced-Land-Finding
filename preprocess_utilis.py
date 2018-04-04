def calibrate_camera(images, nx = 9, ny = 6):
    '''
    Find require parameters to calibrate camera with chessboard images
    param: 
        images: recommending at least 20 images
        nx: number of corners each row
        ny: number of corners each column
    return:
        mtx: camera matrix
        dist: distortion coeficients
    '''
    objp = np.zeros((ny * nx, 3), np.float32)
    objp[:,:2] = np.mgrid[0:nx, 0:ny].T.reshape(-1, 2)
    
    objpoints = []
    imgpoints = []
    for img_path in camera_cal_img_paths:
        img = plt.imread(img_path)
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)
        if ret:
            img = cv2.drawChessboardCorners(img, (nx, ny), corners, ret)
            #plt.imshow(img)
            objpoints.append(objp)
            imgpoints.append(corners)
            
    # camera calibration
    ret, mtx, dist, rvess, tvess = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
    return (mtx, dist)


def threshold_sx_s(image, thres_scheme = 1, s_thres = [120, 255], r_thres = [180,255], g_thres = [200,255],
                  l_thres_shadow = [0, 155], l_thres = [155, 255], sx_thres = [20, 100]):
    '''
    Saturation and gradient of lightness on x direction thresholding the image and generate a binary image
    param:
        image: a image, require color space is RGB
        sx_thres: a list, [lower_bound, higher_bound] threshold for scaled sobel x
        s_thres: a list, [lower_bound, higher_bound] threshold for saturation channel
    return:
        a binary image after applying both thresholding 
    '''
    hls = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
    l_channel = hls[:, :, 1]
    s_channel = hls[:, :, 2]
    r_channel = image[:, :, 0]
    g_channel = image[:, :, 1]
    
    if thres_scheme == 0:
        sobelx = cv2.Sobel(l_channel, cv2.CV_64F, 1, 0)
        abs_sobelx = np.absolute(sobelx)
        scaled_sobel = np.uint8(255 * abs_sobelx / np.max(abs_sobelx))
        sxbinary = np.zeros_like(l_channel)
        sxbinary[(scaled_sobel >= sx_thres[0]) & (scaled_sobel < sx_thres[1])] = 1
        
        region_interest = np.ones_like(l_channel)
        region_interest[480:, 500:700] = 0
        
        combined = np.zeros_like(l_channel)
        combined[(sxbinary == 1) & region_interest == 1] = 1
        return combined

    else:
        r_binary = np.zeros_like(l_channel)
        r_binary[(r_channel >= r_thres[0]) & (r_channel < r_thres[1])] = 1

        g_binary = np.zeros_like(l_channel)
        g_binary[(g_channel >= g_thres[0]) & (g_channel < g_thres[1])] = 1

        s_binary = np.zeros_like(l_channel)
        s_binary[(s_channel >= s_thres[0]) & (s_channel < s_thres[1])] = 1
    
        l_binary_shadow = np.zeros_like(l_channel)
        l_binary_shadow[(l_channel >= l_thres_shadow[0]) & (l_channel < l_thres_shadow[1])] = 1

        l_binary = np.zeros_like(l_channel)
        l_binary[(l_channel >= l_thres[0]) & (l_channel < l_thres[1])] = 1


        combined_binary = np.zeros_like(l_channel)
        combined_binary[( ( (s_binary == 1) & (l_binary == 1) ) |
                      ( (r_binary == 1) & (l_binary_shadow == 1) ) |
                      (g_binary == 1) )] = 1
        return combined_binary


def perspective_transform(src, dest):
    '''
    Find both transform matrix and inverse transform matrix with provided srt and dest
    '''
    M = cv2.getPerspectiveTransform(src, dest)
    M_inv = cv2.getPerspectiveTransform(dest, src)
    return (M, M_inv)