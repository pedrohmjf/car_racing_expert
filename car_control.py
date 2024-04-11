from gym.envs.box2d import CarRacing
from gym.wrappers import TimeLimit, TransformObservation, AutoResetWrapper, RecordVideo
import numpy as np
from image_processing import color_filter, canny, find_middle, pid
from optical_flow import calc_t_matrix

env_video = CarRacing(render_mode="rgb_array", domain_randomize=False)

def always_true(i_epoch):
    return True

video_path = "./videoo"
env_video = TimeLimit(env_video, 5000)
env_video = RecordVideo(env_video, video_path, episode_trigger=always_true)

observation = env_video.reset()
num_steps = 100
action = np.array([0, 1, 0]).astype(np.float32)
previous_error = 0
done = False
i_step = 0
last_frame = None
throttle = 0.1

while not done:
  observation, _, done, truncated, _ = env_video.step(action)
  cropped = observation[:60, :]
  done |= truncated
  try:
    green_image = color_filter(observation)
    canny_image = canny(green_image)
    middle = find_middle(canny_image, previous_error)
  except:
    middle = -15
  if last_frame is not None:
    try:
      flow = calc_t_matrix(cropped, last_frame)
      if flow > 1:
        throttle = 0
      else:
        throttle = 0.03
    except:
      pass
  i_step += 1
  steering = pid(middle, previous_error)
  action = np.array([steering, throttle, 0]).astype(np.float32)
  print(i_step)
  last_frame = cropped