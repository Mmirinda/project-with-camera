import matplotlib.pyplot as plt
import numpy as np
import math

#исходные координаты "человечка"
points = np.array([
    [0, 5, 0],
    [-2, 3, 0],
    [2, 3, 0],
    [-2, 1, -2],
    [2, 1, -2],
    [0, 1, -5],
    [-1, 0, -7],
    [1, 0, -7]
])

angle_degrees = 45
alpha = math.radians(angle_degrees)
cos_alpha = math.cos(alpha)
sin_alpha = math.sin(alpha)
rotated_points = np.zeros_like(points)

for i, (x, y, z) in enumerate(points):
    x_streak = x * cos_alpha - z * sin_alpha
    y_streak = y
    z_streak = x * sin_alpha + z * cos_alpha
    rotated_points[i] = [x_streak, y_streak, z_streak]

fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(points[:, 0], points[:, 1], points[:, 2], c='gray', s=50, label='Исходные')
ax.scatter(rotated_points[:, 0], rotated_points[:, 1], rotated_points[:, 2], c='blue', s=50, label='Повернутые')
connections = [
    (0, 1), (0, 2), (1, 3), (2, 4), (0, 5), (5, 6), (5, 7)
]
for i, j in connections:
    ax.plot([points[i, 0], points[j, 0]], [points[i, 1], points[j, 1]], [points[i, 2], points[j, 2]], 'gray', alpha=0.5)
    ax.plot([rotated_points[i, 0], rotated_points[j, 0]],
            [rotated_points[i, 1], rotated_points[j, 1]],
            [rotated_points[i, 2], rotated_points[j, 2]], 'blue', alpha=0.5)

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title(f'Поворот вокруг оси OY на {angle_degrees}°')
ax.legend()
max_range = np.max([np.max(np.abs(points)), np.max(np.abs(rotated_points))])
ax.set_xlim(-max_range, max_range)
ax.set_ylim(-max_range, max_range)
ax.set_zlim(-max_range, max_range)
plt.show()
