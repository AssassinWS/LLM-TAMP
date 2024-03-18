from scipy.spatial.transform import Rotation
import numpy as np


rot2 = Rotation.from_euler('XYZ', [np.pi, 0, np.pi/3], degrees=False)

# print(rot2.as_rotvec())

# quat = rot2.as_quat()
# print(quat)
o2 ="haha"
msg = []
o1 = True
o2 = o2 + "It's fine"
msg.append(o1)
msg.append(o2)

dict = {}
dict["haha"] = 5
print(dict)

print(msg)