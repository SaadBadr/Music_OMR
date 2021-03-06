########################################################   TEST CELL 1  ##############################################

# # Make test_img copy for testing
# test = row_images.copy()
# test_img = test[1].copy()
# original_row = row_images_original[1].copy()

# # Use the first stave image for testing
# # --------------------------------------

# show_images([test_img], ["Test image before"])


# # # Adjust orientation
# test_img, rotation_angle = adjust_orientation(test_img)
# test_img = np.round(test_img).astype(int)

# original_row = transform.rotate(original_row, rotation_angle, resize=False, clip=False, cval=0)
# original_row= np.round(original_row).astype(int)

# # #Draw vertical and horizontal projection histograms
# draw_vertical_histogram_of_image(test_img)
# draw_horizontal_histogram_of_image(test_img)

# #Truncate the empty left and right sides
# test_img = truncate_left_and_right_empty_spaces(test_img, original_row)
# #Remove the right side bold line
# test_img = remove_right_side_bold_barline(test_img, original_row)



# # # #Detect exactly 5 horizontal lines
# lines = reduce_lines_to_5(detect_lines(test_img))
# draw_lines_on_image(test_img, lines)

# # # Calculate avg spacing of lines and draw detected circles/blobs (Unreliable blob detection)
# draw_circles_on_image(test_img, lines)



# # Draw vertical lines in places where a vertical line exists (vertical histogram value is high)
# fig, axes = plt.subplots(1, 1, figsize=(15, 6))
# axes.imshow(test_img)
# v_lines = get_vertical_lines(test_img, 10)
# for i in range(len(v_lines)):
#     axes.plot([v_lines[i], v_lines[i]], (0, test_img.shape[0]-1), '-b')
# axes.set_title('places where vertical lines exist')
# plt.tight_layout()
# plt.show()


# Draw vertical lines in places where no symbols exist (vertical histogram value is low)
# fig, axes = plt.subplots(1, 1, figsize=(15, 6))
# axes.imshow(test_img)
# vertical_histogram = np.sum(test_img,axis=0)
# thresh = get_most_repeated_pixel_count_in_columns(test_img)
# v_lines = np.where(vertical_histogram <= thresh)[0]

# for i in range(len(v_lines)):
#     axes.plot([v_lines[i], v_lines[i]], (0, test_img.shape[0]-1), '-r')
# axes.set_title('places where no symbols exist')
# plt.tight_layout()
# plt.show()






################################################# TEST CELL 2 #####################################


# test_img_2 = test_img.copy()

# symbols = segment_symbols(test_img_2, original_row)
# fig, axs = plt.subplots(1, len(symbols), figsize=(15,3))
# for i, symbol in enumerate(symbols):
#     axs[i].imshow(symbol)
#     axs[i].set_xticks([]) 
#     axs[i].set_yticks([]) 
# plt.show()



## Dilate lines
#----------------------------------------------------------------------
# after_dilation = dilate_lines(test_img)
# show_images([test_img, after_dilation], ["Before dilation", "Dilated lines"])
# fig, axes = plt.subplots(1, 1, figsize=(15, 6))
# axes.imshow(after_dilation)


## Draw vertical lines where no symbols exist on the dilated lines image
#----------------------------------------------------------------------
# vertical_histogram = np.sum(after_dilation,axis=0)
# thresh = get_most_repeated_pixel_count_in_columns(after_dilation)
# v_lines = np.where(vertical_histogram <= thresh)[0]
# for i in range(len(v_lines)):
#     axes.plot([v_lines[i], v_lines[i]], (0, after_dilation.shape[0]-1), '-r')
# axes.set_title('places where no symbols exist')
# plt.tight_layout()
# plt.show()



########################################################### ROW SEGMENTATION WITH FIGURES ######################

# ## Detect staff lines as a preparation step for stave segmentation.
# ##------------------------------------------------------------------

# linesArr = detect_lines(rotated_image)
# if(len(linesArr) <= 5):

# fig, axes = plt.subplots(figsize=(15, 3))
# axes.imshow(rotated_image, cmap=cm.gray)
# for l in linesArr:
#     axes.plot([l[0][0], l[1][0]], [l[0][1], l[1][1]], '-r')
# axes.set_title('Detected lines')
# plt.tight_layout()
# plt.show()

# ##------------------------------------------------------------------------------------------------------------------------------------
# ##------------------------------------------------------------------------------------------------------------------------------------


# ## Getting the average y values for each line as a preparation step for the next step at which we will segment the different staves.
# ##------------------------------------------------------------------------------------------------------------------------------------

# y_locs = get_avg_lines_heights(linesArr)

# fig, axes = plt.subplots(1, 1, figsize=(15, 3))
# axes.set_title("Before lines adjustment")
# for points in linesArr:
#     axes.plot([points[0][0], points[1][0]], [points[0][1], points[1][1]], '-r')

# fig, axes = plt.subplots(1, 1, figsize=(15, 3))
# axes.set_title("After lines adjustment")
# origin = (0, binary_image.shape[1]-1) #Origin is a tuple that contains (min x value = 0, max x value = img width-1)
# for y in y_locs:
#     axes.plot(origin, (y, y), '-r')

# ##------------------------------------------------------------------------------------------------------------------------------------
# ##------------------------------------------------------------------------------------------------------------------------------------


# ## Getting the mean of the differences between the consecutive lines.
# ## Calculating the threshold by which we will segment the staves (minimum separating distance between the staves).
# ## Segment the image into separate staves
# ##--------------------------------------------------------------------

# sorted_y_locs = sorted(y_locs)
# diffs = []
# mean_diff = 0
# for i in range(len(sorted_y_locs) - 1):
#     diff = sorted_y_locs[i + 1] - sorted_y_locs[i]
#     mean_diff += diff
#     diffs.append(diff)

# mean_diff = mean_diff/(len(sorted_y_locs)-1)
# print("Avg staff line spacing: " + str(mean_diff))

# # Get the minimum separating distance between the staves
# temp = sorted(diffs, reverse=True)
# thres_o = 0
# for i in range(0,len(temp)-1):
#     if(temp[i] > temp [i + 1] * 2):
#         thres_o = temp[i]
#         break
# print("Spacing threshold: " + str(thres_o))

# # Add a segmentation location between any two lines that have a difference greater than or equal to
# # the minimum separating distance between the staves
# seg_locs = []
# for i in range(1, len(sorted_y_locs) - 1):
#     next_diff = sorted_y_locs[i + 1] - sorted_y_locs[i]
#     if next_diff >= (thres_o):
#         seg_locs.append((sorted_y_locs[i + 1] + sorted_y_locs[i]) / 2)

# ## Calculating the maximum difference between the segmentation locations
# max_diff = float('-inf')
# for i in range(len(seg_locs) - 1):
#     diff = seg_locs[i + 1] - seg_locs[i]
#     max_diff = diff if diff > max_diff else max_diff
# ## Using this max diff in calculating the first and last segmentation location
# first_seg_loc = seg_locs[0] - max_diff
# first_seg_loc = first_seg_loc if first_seg_loc >= 0 else 0
# last_seg_loc = seg_locs[-1] + max_diff
# last_seg_loc = last_seg_loc if last_seg_loc >= binary_image.shape[0]-1 else binary_image.shape[0]-1

# seg_locs.insert(0, first_seg_loc)
# # seg_locs = list(filter(lambda x: 0 <= x < binary_image.shape[1], seg_locs))
# seg_locs.append(last_seg_loc)
# fig, axes = plt.subplots(1, 1, figsize=(15, 3))
# axes.imshow(binary_image, cmap=cm.gray)
# for loc in seg_locs:
#     y0, y1 = loc, loc
#     if(loc == 0):
#         y0 = y1 = 2 ## for displaying purposes only
#     axes.plot(origin, (y0, y1), '-r')


# ##------------------------------------------------------------------------------------------------------------------------------------
# ##------------------------------------------------------------------------------------------------------------------------------------


# #row_images= np.zeros(shape= (seg_locs.shape-2, ))
# row_images = []
# row_images_original = []
# for i in range(len(seg_locs) - 1):
#     start_loc, end_loc = (int(round(seg_locs[i])), int(round(seg_locs[i + 1])))
    
#     row = binary_image[start_loc:end_loc, :]
#     row_images.append(row)
#     row = original_image_rotated[start_loc:end_loc, :]
#     row_images_original.append(row)

# # Converting it to numpy array
# row_images = np.array(row_images)
# row_images_original = np.array(row_images_original)


# fig, axes = plt.subplots(len(row_images), 1, figsize=(15, 3))
# for i in range(len(row_images)):
#     axes[i].imshow(row_images[i])

# fig, axes = plt.subplots(len(row_images_original), 1, figsize=(15, 3))
# for i in range(len(row_images_original)):
#     axes[i].imshow(row_images_original[i])








################################## TEST CELL 3 ################################



# import cv2 as cv
# # test_img = (skeletonize(1 - symbols[1][13])*255).astype('uint8')

# # out = test_img.copy()
# test_img = (skeletonize((1 - symbols[0][4]))*255).astype('uint8')

# coords = corner_peaks(corner_harris(test_img), min_distance=2, threshold_rel=0.02)
# coords_subpix = corner_subpix(test_img, coords, window_size=13)

# fig, ax = plt.subplots()
# ax.imshow(test_img, cmap=plt.cm.gray)
# ax.plot(coords[:, 1], coords[:, 0], color='cyan', marker='o',
#         linestyle='None', markersize=6)
# ax.plot(coords_subpix[:, 1], coords_subpix[:, 0], '+r', markersize=15)
# ax.axis((0, 310, 200, 0))
# plt.show()


# # sift = cv.SIFT_create(contrastThreshold=0.09, edgeThreshold=3, sigma=1.6)
# # kp, des = sift.detectAndCompute(test_img,None)
# # test_img=cv.drawKeypoints(test_img,kp,out)
# show_images([test_img])






################################## TEST CELL 4 ################################

# from scipy.ndimage.morphology import binary_fill_holes
# # calc_symbol_position(symbols_no_staff_lines[0][4], "a_2", rows_lines[0][0])
# f = np.array([
#     [0,0,0],
#     [1,1,1],
#     [0,0,0],
# ])
# symbol_cpy = symbols_no_staff_lines[0][4].copy()
# itr = 7
# for i in range(itr):
#     symbol_cpy = binary_dilation(symbol_cpy, selem=f)

# show_images([symbol_cpy], ["symbol after morphology"])
# symbol_cpy = binary_fill_holes(symbol_cpy)
# itr = 9
# for i in range(itr):
#     symbol_cpy = binary_erosion(symbol_cpy, selem=f)



# show_images([symbol_cpy], ["symbol after morphology"])

# lines = rows_lines[0][0]
# diameter = int(get_avg_line_spacing(lines))
# x, y = get_circles(symbol_cpy, diameter, 1)

# min_distance_to_center = float('inf')
# for i, X in enumerate(x):
#     if(abs(X-symbol_cpy.shape[1]//2) < min_distance_to_center):
#         center_x = X
#         center_y = y[i]
#         min_distance_to_center = abs(X-symbol_cpy.shape[1]//2)

# fig, axs = plt.subplots(1, 1, figsize=(20,3))
# axs.imshow(symbol_cpy)
# axs.plot(center_x, center_y, 'o', markeredgecolor='r', markerfacecolor='none', markersize=10)
# plt.show()



################################## TEST CELL 5 ################################

# def check_if_beam_is_on_top(symbol):
# 	topCount = 0
# 	bottomCount = 0
# 	for col in range(symbol.shape[1]):
# 		row = 0
# 		while(row < symbol.shape[0]//2):
# 			if(symbol[row, col] == 1):
# 				topCount+=1
# 				break
# 			row+=1

# 		row = symbol.shape[0]-1
# 		while(row > symbol.shape[0]//2):
# 			if(symbol[row, col] == 1):
# 				bottomCount+=1
# 				break
# 			row-=1
# 	return topCount > bottomCount

# symbol_cpy = symbols_no_staff_lines[1][4].copy()
# lines = rows_lines[1][0]
# diameter = int(get_avg_line_spacing(lines))
# x, y = get_circles(symbol_cpy, diameter, 1)
# b = check_if_beam_is_on_top(symbol_cpy)
# newX = []
# newY = []
# if(b): #beam is on top
#     for i,Y in enumerate(y):
#         if(Y >= symbol_cpy.shape[0]//2):
#             newX.append(x[i])
#             newY.append(Y)
# else: #beam is at bottom
#     for i,Y in enumerate(y):
#         if(Y <= symbol_cpy.shape[0]//2):
#             newX.append(x[i])
#             newY.append(Y)
# newX = np.array(newX)
# newY = np.array(newY)

# fig, axs = plt.subplots(1, 1, figsize=(20,3))
# axs.imshow(symbol_cpy)
# axs.plot(newX, newY, 'o', markeredgecolor='r', markerfacecolor='none', markersize=10)
# plt.show()



############################# TEST CELL 6 ###########################
    # In[5]:


	# for i, row in enumerate(symbols_no_staff_lines):
	# 	first_half_avg_spacing =  int(get_avg_line_spacing(rows_lines[i][0]))
	# 	second_half_avg_spacing =  int(get_avg_line_spacing(rows_lines[i][1]))

	# 	# fig, axs = plt.subplots(1, len(row)+1, figsize=(20,3))
	# 	for j, symbol in enumerate(row):
	# 		diameter = first_half_avg_spacing if j < second_halves_indexes[i] else second_half_avg_spacing
	# 		xArr, yArr = get_circles(symbol, diameter, 1)
	# 		# axs[j].imshow(symbol)
	# 		# axs[j].plot(xArr, yArr, 'o', markeredgecolor='r', markerfacecolor='none', markersize=10)

	# 		selem = np.ones((diameter+1, diameter+1), dtype=np.uint8)
	# 		rr, cc = disk((diameter//2, diameter//2), diameter//2)
	# 		selem[rr,cc] = 1
	# 		# print(selem)
	# 		# axs[-1].imshow(selem)
	# 		# plt.show()