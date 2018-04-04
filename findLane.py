def blind_search(side, binary_warped, undist, nwindows = 9, margin = 100, minpix = 50):
    '''
        blind search for active pixel for the specified lane, if cannot find enough pixel, try another
        thresholding scheme which will only use sobel on x direction
    '''
    # Take a histogram of the bottom half of the image
    histogram = np.sum(binary_warped[binary_warped.shape[0]//2:,:], axis=0)
    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    midpoint = np.int(histogram.shape[0]//2)
    if side == 'left':
        left_lane.current_fit = None
        x_base = np.argmax(histogram[:midpoint])
    else:
        right_lane.current_fit = None
        x_base = np.argmax(histogram[midpoint:]) + midpoint

    # Set height of windows
    window_height = np.int(binary_warped.shape[0]//nwindows)
    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    # Current positions to be updated for each window
    x_current = x_base

    # Create empty lists to receive left and right lane pixel indices
    lane_inds = []

    # Step through the windows one by one
    for window in range(nwindows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = binary_warped.shape[0] - (window+1)*window_height
        win_y_high = binary_warped.shape[0] - window*window_height
        win_x_low = x_current - margin
        win_x_high = x_current + margin

        # Identify the nonzero pixels in x and y within the window
        good_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
        (nonzerox >= win_x_low) &  (nonzerox < win_x_high)).nonzero()[0]


        # Append these indices to the lists
        lane_inds.append(good_inds)

        # If you found > minpix pixels, recenter next window on their mean position
        if len(good_inds) > minpix:
            x_current = np.int(np.mean(nonzerox[good_inds]))

    # Concatenate the arrays of indices
    lane_inds = np.concatenate(lane_inds)

    # Extract left and right line pixel positions
    x = nonzerox[lane_inds]
    y = nonzeroy[lane_inds]

    # Fit a second order polynomial to each
    if len(x) > 50:
        fit = np.polyfit(y, x, 2)

        if side == 'left':
            left_lane.allx = x
            left_lane.ally = y
            left_lane.current_fit = fit
        else:
            right_lane.allx = x
            right_lane.ally = y
            right_lane.current_fit = fit
    else:
        warped = threshold_sx_s(undist, 0)
        findLanes(warped, undist)



def margin_search(side, binary_warped, undist, nwindows = 9, margin = 100, minpix = 50):

    '''
        searching for pixels around the averaged fitted line within certain margin
        Params:
            side:          str, indicates 'left' or 'right'
            binary_warped: np.array, binary value of target image
            undist:        np.array, target image
            nwindows:      int, number of windows vertically need to search
            margin:        int, margin of best fit line set to find active pixel
            minpix:        int, minimun number of existing pixel need to update the current centeriod
    '''
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])

    if side == 'left':
        fit = left_lane.best_fit
    else:
        fit = right_lane.best_fit

    lane_inds = ((nonzerox > (fit[0]*(nonzeroy**2) + fit[1]*nonzeroy +
    fit[2] - margin)) & (nonzerox < (fit[0]*(nonzeroy**2) +
    fit[1]*nonzeroy + fit[2] + margin)))

    # Again, extract left and right line pixel positions
    x = nonzerox[lane_inds]
    y = nonzeroy[lane_inds]

    # Fit a second order polynomial to each
    if len(x) > 50:
        fit = np.polyfit(y, x, 2)
        if side == 'left':
            left_lane.allx = x
            left_lane.ally = y
            left_lane.current_fit = fit
        else:
            right_lane.allx = x
            right_lane.ally = y
            right_lane.current_fit = fit
    else:
        blind_search(side, binary_warped, undist, nwindows, margin, minpix)



def findLanes(binary_warped, undist, nwindows = 9, margin = 100, minpix = 50):
    '''
        Find lanes with margin search if lane was detected in previous frame, otherwise use blind search,
        If lanes found don't seem to parallel and margin search was used, try blind search instead.
        don't quiet parallel with each other use last n averaged fitted lane.
        Params:
            binary_warped: np.array, binary value of target image
            undist:        np.array, target image
            nwindows:      int, number of windows vertically need to search
            margin:        int, margin of best fit line set to find active pixel
            minpix:        int, minimun number of existing pixel need to update the current centeriod
    '''
    ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
    which_test_left = 'blind'
    which_test_right = 'blind'

    if left_lane.detected:
        margin_search('left', binary_warped, undist, nwindows, margin, minpix)
        which_test_left = 'margin'
    else:
        blind_search('left', binary_warped, undist, nwindows, margin, minpix)

    if right_lane.detected:
        margin_search('right', binary_warped, undist, nwindows, margin, minpix)
        which_test_right = 'margin'
    else:
        blind_search('right', binary_warped, undist, nwindows, margin, minpix)


    if which_test_left != 'blind' or which_test_right != 'blind':
        coef_diff = left_lane.current_fit[:-1] - right_lane.current_fit[:-1]
        left_coef_norm = np.linalg.norm(left_lane.current_fit[:-1], 2)
        right_coef_norm = np.linalg.norm(right_lane.current_fit[:-1], 2)
        diff_norm = np.linalg.norm(coef_diff,2) / (left_coef_norm + right_coef_norm)
        coefs_diff_norm.append(diff_norm)
        if diff_norm > 0.1:
            if which_test_left != 'blind':
                blind_search('left', binary_warped, undist, nwindows, margin, minpix)
            if which_test_right != 'blind':
                blind_search('right', binary_warped, undist, nwindows, margin, minpix)


    coef_diff = left_lane.current_fit[:-1] - right_lane.current_fit[:-1]
    left_coef_norm = np.linalg.norm(left_lane.current_fit[:-1], 2)
    right_coef_norm = np.linalg.norm(right_lane.current_fit[:-1], 2)
    diff_norm = np.linalg.norm(coef_diff,2) / (left_coef_norm + right_coef_norm)
    coefs_diff_norm_blind.append(diff_norm)


    if diff_norm < ACCEPTED_THRES:
        left_lane.recent_xfitted.append( left_lane.current_fit[0]*ploty**2 +
                                        left_lane.current_fit[1]*ploty +
                                        left_lane.current_fit[2] )
        left_lane.bestx = np.mean(np.array(list(left_lane.recent_xfitted)), axis = 0)
        left_lane.best_fit = np.polyfit(ploty, left_lane.bestx, 2)
        left_lane.detected = True
        right_lane.recent_xfitted.append( right_lane.current_fit[0]*ploty**2 +
                                         right_lane.current_fit[1]*ploty +
                                         right_lane.current_fit[2] )
        right_lane.bestx = np.mean(np.array(list(right_lane.recent_xfitted)), axis = 0)
        right_lane.best_fit = np.polyfit(ploty, right_lane.bestx, 2)
        right_lane.detected = True

    elif diff_norm > SMOOTH_THRES and left_lane.best_fit is not None and right_lane.best_fit is not None:
        left_lane.current_fit = left_lane.best_fit
        left_lane.allx = left_lane.bestx
        left_lane.ally = ploty
        left_lane.detected = False
        right_lane.current_fit = right_lane.best_fit
        right_lane.allx = right_lane.bestx
        right_lane.ally = ploty
        right_lane.detected = False

    coef_diff = left_lane.current_fit[:-1] - right_lane.current_fit[:-1]
    left_coef_norm = np.linalg.norm(left_lane.current_fit[:-1], 2)
    right_coef_norm = np.linalg.norm(right_lane.current_fit[:-1], 2)
    diff_norm = np.linalg.norm(coef_diff,2) / (left_coef_norm + right_coef_norm)
    coefs_diff_norm_smooth.append(diff_norm)



def projectLane(image, M_inv):

    left_fit = left_lane.current_fit
    right_fit = right_lane.current_fit

    # Create an image to draw the lines on
    warp_zero = np.zeros_like(image[:,:,0]).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

    # Generate x and y values for plotting
    ploty = np.linspace(0, image.shape[0]-1, image.shape[0] )
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))

    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    newwarp = cv2.warpPerspective(color_warp, M_inv, (image.shape[1], image.shape[0]))
    # Combine the result with the original image
    result = cv2.addWeighted(image, 1, newwarp, 0.3, 0)

    return result

def cal_curvature(side, y_eval, h = 20, ym_per_pix = 30/720, xm_per_pix = 3.7/800):

    if side == 'left':
        x, y = left_lane.allx, left_lane.ally
    else:
        x, y = right_lane.allx, right_lane.ally

    # Fit new polynomials to x,y in world space
    fit_cr = np.polyfit(y*ym_per_pix, x*xm_per_pix, 2)

    # Calculate the new radii of curvature
    curverad = ((1 + (2*fit_cr[0]*y_eval*ym_per_pix + fit_cr[1])**2)**1.5) / np.absolute(2*fit_cr[0])

    # Now our radius of curvature is in meters
    lane_x_zero = fit_cr[0] * h**2 + fit_cr[1] * h + fit_cr[2]

    if side == 'left':
        left_lane.lane_x_zero = lane_x_zero
        left_lane.radius_of_curvature = curverad
    else:
        right_lane.lane_x_zero = lane_x_zero
        right_lane.radius_of_curvature = curverad



def add_desc(new_image, num_pix_x, xm_per_pix = 3.7/800):

    lane_center = (right_lane.lane_x_zero - left_lane.lane_x_zero)
    vehicle_center = (num_pix_x * xm_per_pix) / 2

    to_left = lane_center - vehicle_center
    text1 = 'Radius of curvature: {0:.0f}(m)'.format(
    (left_lane.radius_of_curvature + right_lane.radius_of_curvature) / 2)

    side_text = 'left' if to_left > 0 else 'right'

    text2 = 'Vehicle is {:.3f} m {} to the center'.format(abs(to_left), side_text)
    result = new_image.copy()
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(result, text1, (100,100), font, 1, (255,255,255),2, cv2.LINE_AA)
    cv2.putText(result, text2, (100,150), font, 1,(255,255,255), 2, cv2.LINE_AA)

    return result
