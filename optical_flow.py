def calc_t_matrix(frame, last_frame):
  gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
  gray_frame = cv2.normalize(gray_frame, None, 0, 255, cv2.NORM_MINMAX)
  gray_last_frame = cv2.cvtColor(last_frame, cv2.COLOR_BGR2GRAY)
  gray_last_frame = cv2.normalize(gray_last_frame, None, 0, 255, cv2.NORM_MINMAX)
  feature_params = dict( maxCorners=0,
    qualityLevel=0.01,
    minDistance=0.01,
    blockSize=1)
  px_ref = cv2.goodFeaturesToTrack(gray_last_frame, mask = None, **feature_params)
  lk_params = dict( winSize = (15, 15),
    maxLevel = 2,
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
  kp2, st, err = cv2.calcOpticalFlowPyrLK(gray_last_frame, gray_frame, px_ref, None)
  st = st.reshape(st.shape[0])
  old_pts = px_ref[st == 1]
  new_pts = kp2[st == 1]
  normalized_pts = []
  for new_pt, old_pt in zip(new_pts, old_pts):
    normalized_pt = new_pt - old_pt
    normalized_pts.append(normalized_pt)
  normalized_pts = np.array(normalized_pts)

  # (1241.0, 376.0, 718.8560, 718.8560, , )
  H, mask = cv2.findHomography(new_pts, old_pts, method=cv2.RANSAC)
  image_width, image_height, _ = frame.shape
  K = np.array([[1.0, 0.0, (image_width - 1) * 0.5], [0.0, 1.0, (image_height - 1) * 0.5], [0.0, 0.0, 1.0]])
  solutions = cv2.decomposeHomographyMat(H, K)
  for translation in solutions[2][:1]:
    abs_t = (translation[0] ** 2 + translation[1] ** 2 + translation[2] ** 2) ** 0.5
    print(translation)
  return abs_t[0]