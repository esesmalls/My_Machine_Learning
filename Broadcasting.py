import numpy as np
np_array_1=np.random.rand(1,2)
np_array_2=np.random.rand(3,1)
print(f"np_array_1为{np_array_1}")
print(f"np_array_2为{np_array_2}")
#Broadcast_np_array_1=np.broadcast_to(np_array_1,np_array_2.shape)
#print("Broadcast_np_array_1为",Broadcast_np_array_1)
broadcast_np_array_1, broadcast_np_array_2 = np.broadcast_arrays(np_array_1, np_array_2)
print("np_array_1广播后shape:", broadcast_np_array_1.shape)  # (2,3)
print("np_array_2广播后shape:", broadcast_np_array_2.shape)  # (2,3)
#t=np_array_1+np_array_2
#print("t为",t)