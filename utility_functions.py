import numpy as np
import math
#matplotlib
import matplotlib.pyplot as plt
from matplotlib.pyplot import bar
from matplotlib import cm
# Edges
from skimage import data, transform, exposure
from skimage.color import rgb2gray,rgb2hsv
from skimage.filters import median, sobel_h, sobel, sobel_v,roberts, prewitt, threshold_otsu, threshold_niblack, threshold_li, apply_hysteresis_threshold, try_all_threshold
from skimage.feature import canny
from skimage.morphology import binary_erosion, binary_dilation, binary_opening, binary_closing, skeletonize, thin
from skimage.transform import hough_line, hough_line_peaks
from skimage.exposure import histogram
from skimage.draw import line,disk
from skimage.feature import match_template, canny
from scipy.signal import convolve2d
from commonfunctions import *
from scipy.ndimage.morphology import binary_fill_holes





## Utility Functions:
##---------------------------------

## ForSegmentation
##---------------------------------

def adjust_orientation(img):
	skeletonized_image = skeletonize(img) #NOTE this might cause problems if the image is corrupted badly by skeletonization.
	# skeletonized_image = img            #Un comment this line and comment the above line if skeletonization causes any issues.

	tested_angles = np.linspace(-np.pi / 2, np.pi / 2, 360)
	h, theta, d = hough_line(skeletonized_image, theta=tested_angles)

	origin = np.array((0, skeletonized_image.shape[1]))
	angles = []
	for _, angle, dist in zip(*hough_line_peaks(h, theta, d, min_distance=2)):
			angles.append(angle)
	

	values, counts = np.unique(angles, return_counts=True)
	avg_angle = np.rad2deg(values[np.argmax(counts)])
	# print("avg angle: " + str(avg_angle))
	rotation_angle = avg_angle-90 if avg_angle >=0 else 90 + avg_angle
	# print("rotation angle: " + str(rotation_angle))

	rotated_image = img
	# if abs(180 - abs(mean_angle)) > 20:
	# rotated_image = np.round(transform.rotate(img, rotation_angle, resize=True, clip=False, cval=0)).astype(int)
	rotated_image = transform.rotate(rotated_image, rotation_angle, resize=True, clip=False, cval=0, preserve_range=True)
	return rotated_image, rotation_angle

def line_sort_key(line):
	return line[0][1] # return y0

# Returns an array of the detected "horizontal" lines in an input image (sorted by the y component of the first point in each line)
# [ [[x0,y0], [x1, y1]], [[x2, y2], [x3, y3]], ... ] --> Array of lines --> each line is an array of 2 points --> each point is an array of x and y
def detect_lines(img):

	# skeletonized_image = skeletonize(img)
	# skeletonized_image = canny(img)

	skeletonized_image = img



	tested_angles = np.linspace(-np.pi / 2, np.pi / 2, 360)
	h, theta, d = hough_line(skeletonized_image, theta=tested_angles)

	# fig, axes = plt.subplots(figsize=(15, 6))
	# axes.imshow(skeletonized_image, cmap=cm.gray)

	origin = np.array((0, skeletonized_image.shape[1]-1))
	peaks = zip(*hough_line_peaks(h, theta, d, threshold=0.3*np.max(h)))
	

	linesArr = []
	for _, angle, dist in peaks:
			if abs(90 - abs(np.rad2deg(angle)) > 3): # if angle is not close to 90/-90 degress neglect it (we detect horizontal lines only)
					continue
			if(abs(np.rad2deg(angle)) > 0.5): #if angle not equal zero
					y0, y1 = (dist - origin * np.cos(angle)) / np.sin(angle)
			else: #if angle zero
					y0 = 0
					y1 = skeletonized_image.shape[0] - 1
					origin[0] = dist
					origin[1] = dist
			point0 = [origin[0], y0]
			point1 = [origin[1], y1]
			# print(y0, y1)
			if(len([index for index,value in enumerate(linesArr) \
					if \
							( abs(value[0][1] - y0) < 5 and abs(value[1][1] - y1) < 5 ) \
							or ( value[0][1] > y0 and  value[1][1] < y1 ) \
							or ( value[0][1] < y0 and  value[1][1] > y1 ) \
									]) == 0): #If this line nearly approaches or intersects a previous line, don't take it
					linesArr.append([point0, point1])
			else:
					continue
			# axes.plot(origin, (y0, y1), '-r')
			origin[0] = 0
			origin[1] = skeletonized_image.shape[1] - 1

	# axes.set_xlim(origin)
	# axes.set_ylim((skeletonized_image.shape[0], 0))
	# axes.set_axis_off()
	# axes.set_title('Detected lines')
	# plt.tight_layout()
	# plt.show()

	linesArr.sort(key=line_sort_key) #sort by the y component of the first point in each line
	return np.array(linesArr)

# Combines each group of near horizontal lines into one line
def reduce_lines_to_5(lines):
	if(len(lines) <= 5):
			return lines
	Ys = sorted(lines[:,0,1])
	avg_spacing = 0
	for i in range(1, len(Ys)):
			avg_spacing += Ys[i] - Ys[i-1]
	avg_spacing /= (len(Ys) - 1)
	reduced_lines = []
	reduced_lines.append(lines[0])
	for i in range(1, len(lines)):
			if(abs(lines[i][0][1] - lines[i-1][0][1]) >= avg_spacing):
					reduced_lines.append(lines[i])
	return np.array(reduced_lines)

def detect_exactly_5_lines(img):
	return reduce_lines_to_5(detect_lines(img))


def get_avg_lines_heights(linesArr):
	y_locs = [] #y (height) locations
	for line in linesArr:
			y_locs.append((line[0][1] + line[1][1]) / 2)
	return y_locs


def get_avg_line_spacing(linesArr):
	if(len(linesArr) == 1):
		raise Exception('division by zero @ get avg line spacing')
	mean_diff = 0
	for i in range(len(linesArr)-1):
			avgY0 = (linesArr[i][0][1] + linesArr[i][1][1]) / 2
			avgY1 = (linesArr[i+1][0][1] + linesArr[i+1][1][1]) / 2

			diff = avgY1 - avgY0
			mean_diff += diff

	return mean_diff/(len(linesArr)-1)

def remove_staff_lines(img, avg_spacing):
	no_staff_image = img.copy()

	## I tried all the following approaches but i failed to remove staff lines in the non high quality images.

	## Approach 1
	# for i in range(-1,1+1): #Removing the line and removing the two lines above and below it (to eliminate the whole thickness)
	#     rr, cc = line(points[0][1]+i, points[0][0], points[1][1]+i, points[1][0])
	#    # no_staff_image[rr, cc] = 0

	## Approach 2
	# no_staff_image = sobel_v(no_staff_image)
	# thresh = threshold_li(no_staff_image)
	# no_staff_image= no_staff_image < thresh
	# return no_staff_image

	## Approach 3
	vertical_histogram = np.sum(no_staff_image,axis=0)
	unique, counts = np.unique(vertical_histogram, return_counts=True)
	indexOfMaxFrequentVal = np.argmax(counts)
	maxFrequentVal = int(unique[indexOfMaxFrequentVal])
	while(maxFrequentVal <= 5):
	    counts = np.delete(counts, indexOfMaxFrequentVal)
	    unique = np.delete(unique, indexOfMaxFrequentVal)
	    maxFrequentVal = int(unique[np.argmax(counts)])
	no_staff_image[:, np.where(vertical_histogram <= maxFrequentVal+0.2*avg_spacing)] = 0
	# plt.figure()
	# bar(range(0, img.shape[1]), vertical_histogram, width=0.8, align='center')
	return no_staff_image
	

	## Approach 4
	# img = img.copy()
	# Sum = (np.sum(img, axis = 1)).astype(int)
	# maximum = np.max(Sum)
	# hist = np.zeros(img.shape)
	# staff_lines = []
	# for i in range(0, len(Sum)):
	# 	# hist[i, :Sum[i]] = 1
	# 	if Sum[i] >= (maximum - (img.shape[0]//5)):
	# 		staff_lines.append(i)
	# 		img[i] = 0

	# kernel_errosion = np.array([
	# [0, 1, 0],
	# [0, 1, 0],
	# [0, 1, 0]
	# ], 
	# dtype=np.uint8)
	# kernel_close = np.array([
	# [0, 0, 1, 0, 0],
	# [0, 0, 1, 0, 0],
	# [0, 0, 1, 0, 0],
	# [0, 0, 1, 0, 0],
	# [0, 0, 1, 0, 0]
	# ], 
	# dtype=np.uint8)
	# img = binary_erosion(img, selem=kernel_errosion)
	# img = binary_closing(img, selem = kernel_close)
	# return img

def draw_staff_lines(img, linesArr):
	for points in linesArr.astype(int): #looping through the extracted lines
			img_with_staff_lines = img.copy()
			rr, cc = line(points[0][1], points[0][0], points[1][1], points[1][0])
			img_with_staff_lines[rr, cc] = 1
	return img_with_staff_lines

# Gets all vertical lines (on y-axis) that have number of pixels(Equal to one) > certain threshold
def get_vertical_lines(img, thresh=10):
	vertical_histogram = np.sum(img,axis=0)
	return np.where(vertical_histogram > thresh)[0]



# Diameter must be odd number
# Returns the centers of the circles of a given diameter
def get_circles(img, diameter, thresh): 
	selem = np.ones((diameter+1, diameter+1), dtype=np.uint8)
	rr, cc = disk((diameter//2, diameter//2), diameter//2)
	selem[rr,cc] = 1
	img_cpy = img.copy()
	selem = np.array([
		[1,1,1],
		[1,1,1],
		[1,1,1]
	])
	img_cpy = binary_erosion(img_cpy, selem=selem)
	img_cpy = binary_erosion(img_cpy, selem=selem)
	img_cpy = binary_erosion(img_cpy, selem=selem)
	img_cpy = binary_erosion(img_cpy, selem=selem)




	result = convolve2d(np.uint8(img_cpy), selem, mode='same')  + convolve2d(np.uint8(1-img_cpy), 1-selem, mode='same')
	# print(np.max(result), (selem.shape[0] * selem.shape[1]), diameter)
	# result = result / (selem.shape[0] * selem.shape[1])
	
	if(np.max(result) > 0):
		result = result / np.max(result)

	ij = np.where(result >= thresh)
	x, y = ij[::-1]
	# circlesArrX = x
	# circlesArrY = y

	# Combine very close circles to one circle
	circlesArrX = []
	circlesArrY = []
	used = np.zeros(len(x), dtype=np.uint8)
	for i in range(len(x)):
		if(used[i] == 1):
			continue
		used[i] = 1
		accumulator = []
		accumulator.append(i)
		for j in range(len(x)):
			if(used[j] == 1):
				continue
			if abs(x[i] - x[j]) <= diameter and abs(y[i] - y[j]) <= diameter:
				used[j] = 1
				accumulator.append(j)
		centerX = 0
		centerY = 0
		for k in accumulator:
			centerX += x[k]
			centerY += y[k]
		centerX /= len(accumulator)
		centerY /= len(accumulator)
		circlesArrX.append(centerX)
		circlesArrY.append(centerY)

	return np.array(circlesArrX), np.array(circlesArrY)

## Returns 1D array that corresponds to the vertical projection of a binary image
def get_vertical_histogram(img):
	return np.sum(img,axis=0)

## Returns 1D array that corresponds to the horizontal projection of a binary image
def get_horizontal_histogram(img):
	return np.sum(img,axis=1)


## Removes the empty(black) spaces on the left and right sides of a binary image
def truncate_left_and_right_empty_spaces(img, original_img):
	vertical_histogram = get_vertical_histogram(img)
	indixes_of_vertical_histogram_without_zeros = np.where(vertical_histogram > 0)[0]
	
	x = img.shape[1]

	start = indixes_of_vertical_histogram_without_zeros[0]
	while(start < x-1 and np.sum(img[:,start]) <= 5 and np.sum(img[:,start+1]) < 5):	#5 is a threshold
		start += 1
	end = indixes_of_vertical_histogram_without_zeros[-1]
	while(end > 1 and np.sum(img[:,end]) <= 5 and np.sum(img[:,end-1]) < 5):	#5 is a threshold
		end -= 1
	return img[:,start:end], original_img[:,start:end]

def check_if_orientation_is_correct(img):
	y, x = img.shape

	left_sum = 0	
	right_sum = 0
	## Loop through all Ys. and for each Y move from beginning of x untill u find the first pixel that has the value 1
	## If you found this point  before reaching half the image (x/2)  then increment left_sum by 1
	## Do the same but from the other side (Start from the end(x=img.shape[1]-1) and move to the beginning (x=0)) and add to right_sum
	for Y in range(y):
		X = 0
		while(X < x//2):
			if(img[Y, X] == 1):
				left_sum += 1
				break
			X += 1

	for Y in range(y):
		X = x-1
		while(X > x//2):
			if(img[Y, X] == 1):
				right_sum += 1
				break
			X -= 1

	# print(left_sum, right_sum)
	#if left sum is greater than right sum -> the orientation is correct
	return left_sum >= right_sum

def get_most_repeated_pixel_count_in_columns(img):
	vertical_histogram = get_vertical_histogram(img)
	unique, counts = np.unique(vertical_histogram, return_counts=True)
	return unique[np.argmax(counts)]

# Must be done after truncating left and right spaces in a binary image
# Some images does not have this bold barline so this function should not be used for granted although it makes some checks.
def remove_right_side_bold_barline(img, original_img):
	img = img.copy()

	vertical_histogram = get_vertical_histogram(img)
	most_repeated_pixel_count_in_columns = get_most_repeated_pixel_count_in_columns(img)

	x = img.shape[1]-1
	idx = x
	proceed = True
	# while(vertical_histogram[idx] < most_repeated_pixel_count_in_columns): # Move and clear untill you find the bold line
	# 	# img[:,idx] = 0
	# 	idx -= 1
	# 	if(idx <= x - x*10//100): #if you moved 1/10 of the image and you didn't find the bold line then break
	# 		proceed = False

	# if (proceed):
	# 	while(vertical_histogram[idx] > most_repeated_pixel_count_in_columns): # Clear the bold line
	# 		# img[:,idx] = 0
	# 		idx -= 1
	# 	return img[:,:idx] # Return cropped image
	# else:
	# 	return img
	count = 0
	while(count < 2):
		if(vertical_histogram[idx] == most_repeated_pixel_count_in_columns):
			count += 1
		if(idx < x - x*10//100):
			proceed = False
			break
		idx-=1
	if(proceed):
		return img[:, :idx], original_img[:, :idx]
	else:
		return img, original_img


# Must be done after truncating left and right spaces in a binary image
# Some images does not have this bold barline so this function should not be used for granted although it makes some checks.
def remove_left_side_brace(img, original_img):
	img = img.copy()

	vertical_histogram = get_vertical_histogram(img)
	most_repeated_pixel_count_in_columns = get_most_repeated_pixel_count_in_columns(img)

	x = img.shape[1]-1
	idx = 0
	proceed = True
	count = 0
	while(count < 5):
		if(vertical_histogram[idx] == most_repeated_pixel_count_in_columns):
			count += 1
		if(idx >= x - x*10//100):
			proceed = False
			break
		idx+=1
	if(proceed):
		return img[:, idx:], original_img[:, idx:]
	else:
		return img, original_img
	
def draw_vertical_histogram_of_image(img):
	vertical_histogram = np.sum(img,axis=0)
	fig, axes = plt.subplots(1, 1)
	axes.bar(range(0,len(vertical_histogram)),vertical_histogram.astype(np.uint8), width=0.8, align='center')
	axes.set_title('Vertical histogram')
	plt.tight_layout()
	plt.show()

def draw_horizontal_histogram_of_image(img):
	horizontal_histogram = np.sum(img,axis=1)
	fig, axes = plt.subplots(1, 1)
	axes.barh(range(0,len(horizontal_histogram)),horizontal_histogram.astype(np.uint8), height=0.8, align='center')
	axes.set_title('Horizontal histogram')
	plt.tight_layout()
	plt.show()

def draw_lines_on_image(img, linesArr):
	fig, axes = plt.subplots(figsize=(15, 6))
	axes.imshow(img, cmap=cm.gray)
	for l in linesArr:
		axes.plot([l[0][0], l[1][0]], [l[0][1], l[1][1]], '-r')
	axes.set_title('Detected lines')
	plt.tight_layout()
	plt.show()


## Takes lines array (must be the reduced 5 lines from which it will calculate the average spacing)
def draw_circles_on_image(img, lines):
	fig, axes = plt.subplots(1, 1, figsize=(15, 6))
	axes.imshow(img)

	avg_spacing = int(get_avg_line_spacing(lines)) - 1 #Subtracting 1 for tolerance
	if(avg_spacing%2 == 0):
		avg_spacing -= 1
	# print("avg spacing: ", avg_spacing)
	circlesArrX, circlesArrY = get_circles(img, avg_spacing)
	# print(sorted(circlesArrY[circlesArrX < 500]))
	# print(lines[:,0,1])
	axes.plot(circlesArrX, circlesArrY, 'o', markeredgecolor='r', markerfacecolor='none', markersize=10)
	axes.set_title('Detected circles')
	plt.tight_layout()
	plt.show()


def dilate_lines(img):
	filter_1 = np.array([
	[0, 0, 0, 0, 0],
	[0, 0, 0, 0, 0],
	[0, 1, 1, 1, 0],
	[0, 0, 0, 0, 0],
	[0, 0, 0, 0, 0]
	])
	filter_2 = np.array([
	[0, 0, 0, 0, 0],
	[0, 0, 0, 0, 0],
	[1, 0, 0, 0, 1],
	[0, 0, 0, 0, 0],
	[0, 0, 0, 0, 0]
	])
	filter_3 = np.array([
	[0, 0, 0, 0, 0, 0, 0],
	[0, 0, 0, 0, 0, 0, 0],
	[0, 0, 0, 0, 0, 0, 0],
	[1, 0, 0, 0, 0, 0, 1],
	[0, 0, 0, 0, 0, 0, 0],
	[0, 0, 0, 0, 0, 0, 0],
	[0, 0, 0, 0, 0, 0, 0]
	])
	filter_4 = np.array([
	[0, 0, 1, 0, 0],
	[0, 0, 0, 0, 0],
	[0, 0, 0, 0, 0],
	[0, 0, 0, 0, 0],
	[0, 0, 1, 0, 0]
	])
	filter_5 = np.array([
	[0, 0, 0, 0, 0],
	[0, 0, 0, 0, 0],
	[0, 0, 0, 0, 0],
	[0, 0, 0, 0, 0],
	[0, 0, 0, 0, 0]
	])
	# & (1-binary_erosion(img, selem=filter_4)
	for i in range(3):
		img = binary_dilation(img, selem=filter_1)
	return img

def segment_image_into_rows(binary_image, original_image):

	## Detect staff lines as a preparation step for stave segmentation.
	##------------------------------------------------------------------
	linesArr = detect_lines(binary_image)
	# if the number of lines detected is less than or equal to 5 then the 
	# img contains only one stave/row.
	if(len(linesArr) <= 8):
		return [binary_image], [original_image], [0, binary_image.shape[0]-1]


	## Getting the average y values for each line as a preparation step 
	## for the next step at which we will segment the different staves.
	##-------------------------------------------------------------------

	y_locs = get_avg_lines_heights(linesArr)


	## Getting the mean of the differences between the consecutive lines.
	## Calculating the threshold by which we will segment the staves 
	## (minimum separating distance between the staves).
	## Segment the image into separate staves.
	##--------------------------------------------------------------------
	sorted_y_locs = sorted(y_locs)
	diffs = []
	mean_diff = 0
	for i in range(len(sorted_y_locs) - 1):
		diff = sorted_y_locs[i + 1] - sorted_y_locs[i]
		mean_diff += diff
		diffs.append(diff)
	mean_diff = mean_diff/(len(sorted_y_locs)-1)

	# Get the minimum separating distance between the staves.
	temp = sorted(diffs, reverse=True)
	thres_o = 0
	for i in range(0,len(temp)-1):
		if(temp[i] > temp [i + 1] * 2):
			thres_o = temp[i]
			break

	# Add a segmentation location between any two lines that have 
	# a difference greater than or equal to the minimum separating
	# distance between the staves.
	seg_locs = []
	for i in range(1, len(sorted_y_locs) - 1):
		next_diff = sorted_y_locs[i + 1] - sorted_y_locs[i]
		if next_diff >= (thres_o):
			seg_locs.append((sorted_y_locs[i + 1] + sorted_y_locs[i]) / 2)

	## Calculating the maximum difference between the segmentation locations.
	max_diff = float('-inf')
	for i in range(len(seg_locs) - 1):
		diff = seg_locs[i + 1] - seg_locs[i]
		max_diff = diff if diff > max_diff else max_diff

	## Using this max diff in calculating the first and last segmentation location.
	first_seg_loc = seg_locs[0] - max_diff
	first_seg_loc = first_seg_loc if first_seg_loc >= 0 else 0
	last_seg_loc = seg_locs[-1] + max_diff
	last_seg_loc = last_seg_loc if last_seg_loc < binary_image.shape[0]-1 else binary_image.shape[0]-1
	seg_locs.insert(0, first_seg_loc)
	seg_locs.append(last_seg_loc)

	row_images = []
	row_images_original = []
	for i in range(len(seg_locs) - 1):
		start_loc, end_loc = (int(round(seg_locs[i])), int(round(seg_locs[i + 1])))
		row = binary_image[start_loc:end_loc, :]
		row_images.append(row)
		row = original_image[start_loc:end_loc, :]
		row_images_original.append(row)

	return np.array(row_images, dtype=object), np.array(row_images_original, dtype=object), seg_locs



def segment_symbols(binary_img, original_img, is_first_half):
	# Draw vertical lines in places where no symbols exist (vertical histogram value is low)
	binary_img, original_img = truncate_left_and_right_empty_spaces(binary_img, original_img)
	if(not is_first_half):
		binary_img, original_img = remove_right_side_bold_barline(binary_img, original_img)
	if(is_first_half):
		binary_img, original_img = remove_left_side_brace(binary_img, original_img)
	
	linesArr = reduce_lines_to_5(detect_lines(binary_img))
	avg_spacing = int(get_avg_line_spacing(linesArr))

	f1 = np.array([
	[0,1,0],
	[0,1,0],
	[0,1,0]
	])
	f2 = np.array([
	[0,0,0],
	[1,1,0],
	[0,0,0]
	])

	binary_img = remove_staff_lines(binary_img, avg_spacing)
	binary_img = binary_dilation(binary_img, selem=f2)

	# show_images([binary_img])


	number_of_itrs = 1
	for i in range(number_of_itrs):
		binary_img = binary_opening(binary_img, selem=f1)

	binary_img = binary_opening(binary_img, selem=f2)


	# show_images([binary_img])


	############# SEGMENTATION ############
	vertical_histogram = get_vertical_histogram(binary_img)
	thresh = get_most_repeated_pixel_count_in_columns(binary_img)
	v_lines = np.where(vertical_histogram <= thresh)[0]

	symbols_locs = []
	
	for i in range(len(v_lines)-1):
		x_loc = v_lines[i]
		next_x_loc = v_lines[i+1]
		if(next_x_loc != x_loc+1): #  and next_x_loc - x_loc >= avg_spacing-avg_spacing*50//100
			start = x_loc-2 if x_loc > 2 else x_loc
			end = next_x_loc+2 if next_x_loc <= binary_img.shape[1]-1 else binary_img.shape[1]-1
			symbols_locs.append([start,end])

	symbols_locs = np.array(symbols_locs)
	symbols= []
	symbols_no_staff_lines= []

	for symbol_loc in symbols_locs:
		symbols.append(original_img[:, symbol_loc[0]:symbol_loc[1]])	
		symbol_no_staff_lines = binary_img[:, symbol_loc[0]:symbol_loc[1]]
		symbol_no_staff_lines = binary_erosion(symbol_no_staff_lines, selem=f1)
		symbol_no_staff_lines = binary_erosion(symbol_no_staff_lines, selem=f1)
		symbol_no_staff_lines = binary_erosion(symbol_no_staff_lines, selem=f1)

		symbol_no_staff_lines = binary_dilation(symbol_no_staff_lines, selem=f1)
		symbol_no_staff_lines = binary_dilation(symbol_no_staff_lines, selem=f1)
		symbol_no_staff_lines = binary_dilation(symbol_no_staff_lines, selem=f1)

		symbols_no_staff_lines.append(symbol_no_staff_lines)

	return symbols, symbols_no_staff_lines, linesArr


def check_if_beam_is_on_top(symbol):
	topCount = 0
	bottomCount = 0
	for col in range(symbol.shape[1]):
		row = 0
		while(row < symbol.shape[0]//2):
			if(symbol[row, col] == 1):
				topCount+=1
				break
			row+=1

		row = symbol.shape[0]-1
		while(row > symbol.shape[0]//2):
			if(symbol[row, col] == 1):
				bottomCount+=1
				break
			row-=1
	return topCount > bottomCount
##----------------------------------------------------------------------------------------------------------------------------------------------

## For Classification
##------------------

def classify_symbol(symbol):
	# TODO
	return "<Symbol X>"


def calc_symbol_position(symbol, label, lines):
	if(not (label in ["a_1", "a_2", "a_4", "a_8", "a_16", "a_32", "b_8", "b_16", "b_32", "chord"])):
		return {"label": label, "centers": []}

	if(label in ["a_1", "a_2"]):
		f = np.array([
    [0,0,0],
    [1,1,1],
    [0,0,0],
		])
		symbol_cpy = symbol.copy()
		itr = 7
		for i in range(itr):
			symbol_cpy = binary_dilation(symbol_cpy, selem=f)
		symbol_cpy = binary_fill_holes(symbol_cpy)
		itr = 9
		for i in range(itr):
			symbol_cpy = binary_erosion(symbol_cpy, selem=f)

		diameter = int(get_avg_line_spacing(lines))
		x, y = get_circles(symbol_cpy, diameter, 1)
		min_distance_to_center = float('inf')
		for i, X in enumerate(x):
			if(abs(X-symbol_cpy.shape[1]//2) < min_distance_to_center):
				center_x = X
				center_y = y[i]
				min_distance_to_center = abs(X-symbol_cpy.shape[1]//2)
		return {"label": label, "centers": [center_y]}


	diameter = int(get_avg_line_spacing(lines))
	x, y = get_circles(symbol, diameter, 1)

	crotchets_positions = np.array(y)

	if(label in ["b_8", "b_16", "b_32"]):

		# b = check_if_beam_is_on_top(symbol)
		# newX = []
		# newY = []
		# if(b): #beam is on top
		# 		for i,Y in enumerate(y):
		# 				if(Y >= symbol_cpy.shape[0]//2):
		# 						newX.append(x[i])
		# 						newY.append(Y)
		# else: #beam is at bottom
		# 		for i,Y in enumerate(y):
		# 				if(Y <= symbol_cpy.shape[0]//2):
		# 						newX.append(x[i])
		# 						newY.append(Y)
		# newX = np.array(newX)
		# newY = np.array(newY)
		# crotchets_positions = newY.copy()

		if(check_if_beam_is_on_top(symbol)):
			crotchets_positions =  crotchets_positions[np.where(crotchets_positions >= symbol.shape[0]//2)]
		else:
			crotchets_positions =  crotchets_positions[np.where(crotchets_positions <= symbol.shape[0]//2)]

	return {"label": label, "centers": crotchets_positions}
	# lines_locs = lines[:,1]

	# pitches = {
	# 	"c": lines_locs[4] + diameter,			#c
	# 	"d": lines_locs[4] + diameter//2,		#d
	# 	"e": lines_locs[4],									#e
	# 	"f": lines_locs[3] + diameter//2,		#f
	# 	"g": lines_locs[3],									#g
	# 	"a": lines_locs[2] + diameter//2,		#a
	# 	"b": lines_locs[2],									#b
	# 	"c2": lines_locs[1] + diameter//2,	#c2
	# 	"d2": lines_locs[1],								#d2
	# 	"f2": lines_locs[0] + diameter//2,	#f2
	# 	"g2": lines_locs[0],								#g2
	# 	"a2": lines_locs[0] - diameter//2,	#a2
	# 	"b2": lines_locs[0] - diameter,			#b2
	# }
	
	# symbols_pitches = []
	# for crotchet_position in crotchets_positions:
	# 	nearest_pitch_key = 'c'
	# 	min_distance = float('inf')
	# 	for pitch_key in pitches.keys():
	# 		if(abs(pitches[pitch_key] - crotchet_position) < min_distance):
	# 			nearest_pitch_key = pitch_key
	# 			min_distance = abs(pitches[pitch_key] - crotchet_position)

	# 	symbols_pitches.append(nearest_pitch_key)
	
	# return symbols_pitches
		
