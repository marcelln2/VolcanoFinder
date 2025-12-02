import os
import time
import airsim
import numpy as np

VEHICLE = "Satellite"
OUTPUT_FOLDER = "captured_images"

client = airsim.MultirotorClient()
client.confirmConnection()
client.enableApiControl(True, vehicle_name=VEHICLE)
client.armDisarm(True, vehicle_name=VEHICLE)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

def save_image(step_id):
    responses = client.simGetImages([
        airsim.ImageRequest("0", airsim.ImageType.Scene, False, False)
    ], vehicle_name=VEHICLE)
    resp = responses[0]
    if resp.image_data_uint8:
        img_1d = np.frombuffer(resp.image_data_uint8, dtype=np.uint8)
        img_rgb = img_1d.reshape(resp.height, resp.width, 3)
        filename = os.path.join(OUTPUT_FOLDER, f"{step_id + 1}.png")
        airsim.write_png(filename, img_rgb)


R = 387
steps = 38
A = 1 / (R * 1.25)
y_increment = R / steps

c_yaw = 180

for step in range(steps + 1):
    x = y_increment * step
    y = A * (x ** 2)
    z = -1.0

    client.moveToPositionAsync(x, y, z, 5,
                             yaw_mode=airsim.YawMode(is_rate=False, yaw_or_rate=c_yaw),
                            vehicle_name=VEHICLE).join()
    c_yaw += 2.36842105263
    if step % 8 == 0:
        save_image(step)
    time.sleep(0.1)

client.hoverAsync(vehicle_name=VEHICLE).join()
print('Surveying finished!')