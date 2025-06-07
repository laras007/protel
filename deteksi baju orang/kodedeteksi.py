import cv2
import numpy as np

# Load image
image = cv2.imread("orang2.jpg")
image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)


# Menentukan batas dan target warna
# Kaos hijau
lower_shirt = np.array([35, 50, 50])
upper_shirt = np.array([85, 255, 255])
green_target = np.array([50, 130, 60])  # Target color for green shirt

# celana biru dongker
lower_pants = np.array([90, 50, 50])
upper_pants = np.array([130, 255, 255])
dark_blue_target = np.array([100, 0, 0])  # Dark blue in BGR

# hijau muda kulit
lower_skin = np.array([0, 20, 150])
upper_skin = np.array([30, 255, 255])
light_green_target = np.array([50, 200, 150])  # Light green for skin

# kernel operasi morphological 
kernel = np.ones((1, 1), np.uint8)

# Gradien morphological untuk edge detection
edges = cv2.morphologyEx(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), cv2.MORPH_GRADIENT, kernel)
edges_colored = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

# transformasi berdasarkan mask
# Kaos
mask_shirt = cv2.inRange(image_hsv, lower_shirt, upper_shirt)
edges_colored[mask_shirt > 0] = green_target

# celana
mask_pants = cv2.inRange(image_hsv, lower_pants, upper_pants)
edges_colored[mask_pants > 0] = dark_blue_target

# kulit
mask_skin = cv2.inRange(image_hsv, lower_skin, upper_skin)
edges_colored[mask_skin > 0] = light_green_target

#menampilkan gambar
combined_image = np.hstack((image, edges_colored))
cv2.imshow('Before and After', combined_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
