import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
import os

####################################
############ CONSTANTS #############
####################################
thresh_min      = 20
thresh_max      = 100
s_thresh_min    = 170
s_thresh_max    = 255
ym_per_pix      = 30/720    
xm_per_pix      = 3.7/700   

def image_show(img,text):
    f, ax1 = plt.subplots(1, 1)
    ax1.imshow(img,cmap='gray')
    ax1.set_title(text, fontsize=15)
    plt.show()
def get_binary(img,min,max):
    binary = np.zeros_like(img)
    binary[(img >= min) & (img <= max)] = 1
    return binary
def define_vertices(img,x,y): 
    imshape = img.shape
    xcenter = imshape[1]/2
    return np.float32([(0,imshape[0]),(xcenter-x, y), (xcenter+x,y), (imshape[1],imshape[0])])
def undistort(image, mtx, dist):
    image = cv2.undistort(image, mtx, dist, None, mtx)
    return image
def cam_calibration(directory, nx, ny, img_size,save_file = True): 
    images = os.listdir(directory)      
    objp = np.zeros((nx*ny,3), np.float32)
    objp[:,:2] = np.mgrid[0:nx, 0:ny].T.reshape(-1,2)

    objpoints = [] 
    imgpoints = []
    
    for img in images:
        gray = cv2.imread(directory+img,cv2.IMREAD_GRAYSCALE)
        ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)
    
        if ret == True:
            objpoints.append(objp)
            imgpoints.append(corners)
            
     
                
            
    if (len(objpoints) == 0 or len(imgpoints) == 0):
        raise Error("Calibration Failed")
            
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_size, None, None)
    
    return mtx, dist

def grad_and_color_thresh(img,img_name='',imgshow=True,save_file=False):
    gray            = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
    s_channel       = cv2.cvtColor(img,cv2.COLOR_RGB2HLS)[:,:,2]

    abs_sobelx      = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 1, 0))
    scaled_sobel    = np.uint8(255*abs_sobelx/np.max(abs_sobelx))

    binary_sobel_x  = get_binary(scaled_sobel,thresh_min,thresh_max)
    binary_hls_s    = get_binary(s_channel,s_thresh_min,s_thresh_max)

    combined_binary = np.zeros_like(binary_sobel_x)
    combined_binary[(binary_hls_s == 1) | (binary_sobel_x == 1)] = 1


    return combined_binary
def transform(undist,vertices):
    img_size      = (undist.shape[1], undist.shape[0])
    new_top_left  = np.array([vertices[0,0],0])
    new_top_right = np.array([vertices[3,0],0])
    
    offset        = [90,10]
    dst           = np.float32([vertices[0]+offset,new_top_left+offset,new_top_right-offset ,vertices[3]-offset])    
    
    M           = cv2.getPerspectiveTransform(vertices, dst)
    Minv        = cv2.getPerspectiveTransform(dst, vertices) 
    warped      = cv2.warpPerspective(undist, M, img_size, flags=cv2.INTER_LINEAR)
    return warped,Minv
def get_lines(img,windows_number,im_show = True):
    margin        = 85
    minpix        = 55
      
    l_lane_inds   = []
    r_lane_inds   = []

    out_img       = np.dstack((img, img, img))*255
    ploty         = np.linspace(0, img.shape[0]-1, img.shape[0] )
    nonzero       = img.nonzero()
    nonzeroy      = np.array(nonzero[0])
    nonzerox      = np.array(nonzero[1])
    
    window_height = np.int(img.shape[0]/windows_number)
    
    histogram     = np.sum(img[img.shape[0]//2:,:], axis=0)
    center        = np.int(histogram.shape[0]/2)
    left_line     = np.argmax(histogram[:center])
    right_line    = np.argmax(histogram[center:]) + center
   
    lx_current    = left_line
    rx_current    = right_line
  
    for window in range(windows_number):
        win_y_low       = img.shape[0]   - (window+1)*window_height
        win_y_high      = img.shape[0]   - window*window_height
        win_xleft_low   = lx_current     - margin
        win_xleft_high  = lx_current     + margin
        win_xright_low  = rx_current     - margin
        win_xright_high = rx_current     + margin

        good_left_inds  = ((nonzeroy >= win_y_low)      & 
                           (nonzeroy < win_y_high)      & 
                           (nonzerox >= win_xleft_low)  & 
                           (nonzerox < win_xleft_high)).nonzero()[0]
        
        good_right_inds = ((nonzeroy >= win_y_low)      & 
                           (nonzeroy < win_y_high)      & 
                           (nonzerox >= win_xright_low) &
                           (nonzerox < win_xright_high)).nonzero()[0]
        
        l_lane_inds.append(good_left_inds)
        r_lane_inds.append(good_right_inds)

        if len(good_left_inds) > minpix:
            lx_current = np.int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:        
            rx_current = np.int(np.mean(nonzerox[good_right_inds]))
            
    l_lane_inds = np.concatenate(l_lane_inds)
    r_lane_inds = np.concatenate(r_lane_inds)  

    left_fit    = np.polyfit(nonzeroy[l_lane_inds] , nonzerox[l_lane_inds], 2)  
    right_fit   = np.polyfit(nonzeroy[r_lane_inds] , nonzerox[r_lane_inds], 2)
    
    left_fitx   = left_fit [0]*ploty**2 + left_fit [1]*ploty + left_fit[2]
    right_fitx  = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

    out_img[nonzeroy[l_lane_inds], nonzerox[l_lane_inds]] = [255, 0, 0]
    out_img[nonzeroy[r_lane_inds], nonzerox[r_lane_inds]] = [0, 0, 255]


    
    return l_lane_inds,r_lane_inds,ploty
def get_x_for_line(line_fit, line_y):
    poly = np.poly1d(line_fit)
    return poly(line_y)
def get_curvature_center(warped,left,right,ploty):
   
    nonzeroy            = np.array(warped[0])
    nonzerox            = np.array(warped[1])
    y_eval              = np.max(ploty)

    left_fit_cr         = np.polyfit(nonzeroy[left]*ym_per_pix, nonzerox[left]*xm_per_pix, 2)
    right_fit_cr        = np.polyfit(nonzeroy[right]*ym_per_pix, nonzerox[right]*xm_per_pix, 2)
    
    left_fitx_bottom_m  = get_x_for_line(left_fit_cr, 720 * ym_per_pix)
    right_fitx_bottom_m = get_x_for_line(right_fit_cr, 720 * ym_per_pix)
    
    left_curvature      = ((1 + (2*left_fit_cr[0]*y_eval*ym_per_pix + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
    right_curvature     = ((1 + (2*right_fit_cr[0]*y_eval*ym_per_pix + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])
    
    center_ideal_m      = 1280 * xm_per_pix / 2

    center_actual_m     = np.mean([left_fitx_bottom_m, right_fitx_bottom_m])
    distance_from_center= abs(center_ideal_m - center_actual_m)
    return left_curvature,right_curvature,distance_from_center
def draw_region(warped,left,right,Minv,undisorted,ploty,im_show=True):

    warp_zero = np.zeros_like(warped).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

    nonzero       = warped.nonzero()
    nonzeroy      = np.array(nonzero[0])
    nonzerox      = np.array(nonzero[1])
    
    left_fit    = np.polyfit(nonzeroy[left] , nonzerox[left], 2)  
    right_fit   = np.polyfit(nonzeroy[right] , nonzerox[right], 2)
    
    left_fitx   = left_fit [0]*ploty**2 + left_fit [1]*ploty + left_fit[2]
    right_fitx  = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

    pts_left    = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right   = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts         = np.hstack((pts_left, pts_right))

    cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))
    newwarp     = cv2.warpPerspective(color_warp, Minv, (undisorted.shape[1], undisorted.shape[0])) 
    result      = cv2.addWeighted(undisorted, 1, newwarp, 0.3, 0)
    if(im_show==True):
        plt.imshow(result)
        plt.show()
    return result

def calibrate():

   mtx, dist = cam_calibration('camera_cal/', 9, 6, (720, 1280))
   return mtx,dist

def draw_line(img,mtx,dist):
    undisorted                     = undistort(img, mtx, dist)
    combined_binary                = grad_and_color_thresh(undisorted,False,False)
    warped,Minv                    = transform(combined_binary,np.float32([[190,720],[589,457],[698,457],[1145,720]]))
    left,right,ploty               = get_lines(warped,9,False)
    l_curv,r_curv,d_center         = get_curvature_center(warped.nonzero(),left,right,ploty)
    result                         = draw_region(warped,left,right,Minv,undisorted,ploty,False)
     
    cv2.putText(result,'Radius of Left Curvature:  %.2fm' % l_curv,(20,40), cv2.FONT_HERSHEY_SIMPLEX, 1,(255,255,255),2)
    cv2.putText(result,'Radius of Right Curvature: %.2fm' % r_curv,(20,80), cv2.FONT_HERSHEY_SIMPLEX, 1,(255,255,255),2)
    cv2.putText(result,'Distance from Center:      %.2fm' % d_center,(20,120), cv2.FONT_HERSHEY_SIMPLEX, 1,(255,255,255),2)
    return result




