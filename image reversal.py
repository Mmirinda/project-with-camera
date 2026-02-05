import cv2
import numpy as np
from matplotlib import cm 
import open3d as o3d

# Шаг 1: Загрузка исходного изображения
image_color = cv2.imread('D:/images for test realsence/my_hend_realsence.png')  # замените на путь к вашему файлу
cv2.imshow('the first', image_color)
# Шаг 2: Конвертация из BGR (OpenCV) в RGB для работы с colormap
rgb = cv2.cvtColor(image_color, cv2.COLOR_BGR2RGB)
""" cv2.imshow('the next', rgb)
cv2.waitKey(0)
cv2.destroyAllWindows() """


# Шаг 3: Создаём обратную карту цветов для палитры Jet (или аналогичной)
jet_cmap = cm.get_cmap('jet')  # Jet: синий (низкие значения) → красный (высокие)
gray_values = np.arange(256, dtype=np.uint8)  # глубина 0–255
color_values = jet_cmap(gray_values, bytes=True)[:, :3]  # RGB без альфа-канала

# Словарь: цвет → значение глубины (0–255)
color_to_gray_map = dict(zip(map(tuple, color_values), gray_values))

# Шаг 4: Преобразуем цвет каждого пикселя в значение глубины
def pixel_to_depth(pixel):
    return color_to_gray_map.get(tuple(pixel), 0)  # если цвет не найден → глубина 0

depth_gray = np.apply_along_axis(pixel_to_depth, 2, rgb)

# Шаг 5: Учитываем особенность: чёрный = >4 м (максимальное расстояние)
# Предполагаем, что чёрный (0,0,0) соответствует максимальной глубине (>4 м)
# Задаём порог: если пиксель близок к чёрному → присваиваем максимальное значение (255)
black_threshold = 30  # порог для определения «чёрного» (можно настроить)
is_black = np.all(rgb < black_threshold, axis=2)
depth_gray[is_black] = 255  # максимальная глубина (>4 м)

# Шаг 6: Инвертируем значения (синий = ближний → белый, красный = дальний → чёрный)
depth_gray = 255 - depth_gray

# Шаг 7: Нормализация (на всякий случай) и конвертация в uint8
depth_gray = cv2.normalize(depth_gray, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

# Шаг 8: Сохранение результата
cv2.imwrite('depth_grayscale.png', depth_gray)

# Шаг 9: Визуализация
""" cv2.imshow('Исходное (цветное)', image_color)
cv2.imshow('Преобразованное (grayscale)', depth_gray)
cv2.waitKey(0)
cv2.destroyAllWindows()
 """
# Загрузка глубины для 3D-расчётов
depth_image = cv2.imread('depth_grayscale.png', cv2.IMREAD_UNCHANGED)
depth_meters = depth_image.astype(np.float32) / 255.0 * 4.0  # в метры (0–4м)

# Вычисляем 3D-координаты (шаги из первого фрагмента)
height, width = depth_meters.shape
x, y = np.meshgrid(np.arange(width), np.arange(height))
fx, fy = 615, 615
cx, cy = width / 2, height / 2
X = (x - cx) * depth_meters / fx
Y = (y - cy) * depth_meters / fy
Z = depth_meters

# Формируем облако точек Open3D
points = np.stack([X, Y, Z], axis=-1).reshape(-1, 3)
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(points)

# Добавляем цвет (если нужно)
rgb_flat = rgb.reshape(-1, 3)
pcd.colors = o3d.utility.Vector3dVector(rgb_flat / 255.0)

# Визуализируем
o3d.visualization.draw_geometries([pcd])
cv2.waitKey(0)
cv2.destroyAllWindows()

# Сохраняем (опционально)
""" o3d.io.write_point_cloud("cloud.ply", pcd) """