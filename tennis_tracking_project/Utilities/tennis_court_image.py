import numpy as np
import matplotlib.pyplot as plt

# def court_2D_image(Corner_points,save_path):

#     Arena_area = 3 # meters extra area around the court
#     Court_Width = 8.23
#     Court_Length = 23.77
#     Center_line = Court_Length / 2

#     Corner_points  = np.array(Corner_points)
#     x = Corner_points[:, 0]  # width (meters)
#     y = Corner_points[:, 1]  # length (meters)

#         # Tennis court dimensions in meters
#     court_outline = np.array([
#         [0, 0],
#         [Court_Width , 0],
#         [Court_Width, Court_Length],
#         [0, Court_Length],
#         [0, 0]
#     ])

#     Court_center_line = np.array([
#         [0, Center_line],
#         [Court_Width, Center_line]
#     ])

#     plt.figure(figsize=(5, 10))

#     # Court boundary
#     plt.plot(court_outline[:, 0],court_outline[:, 1],color='black',linewidth=2)
#     # Radar / player points
#     plt.scatter(x, y, c='red', s=80)
#     # Center (halfway) line
#     plt.plot(Court_center_line[:, 0],Court_center_line[:, 1],color='black',linewidth=2)

#     plt.title("Radar")
#     plt.gca().set_aspect('equal', adjustable='box')
#     plt.grid(False)
#     plt.xlim(-Arena_area, 8.23 + Arena_area)
#     plt.ylim(-Arena_area, 23.77 + Arena_area)
#     plt.xlabel("Width (meters)")
#     plt.ylabel("Length (meters)")
#     plt.savefig(save_path, dpi=200, bbox_inches="tight")
#     plt.close()

def court_2D_image(Corner_points):
    Arena_area = 3
    Court_Width = 8.23
    Court_Length = 23.77
    Center_line = Court_Length / 2

    Corner_points = np.array(Corner_points)
    x = Corner_points[:, 0]
    y = Corner_points[:, 1]

    court_outline = np.array([
        [0, 0],
        [Court_Width, 0],
        [Court_Width, Court_Length],
        [0, Court_Length],
        [0, 0]
    ])

    Court_center_line = np.array([
        [0, Center_line],
        [Court_Width, Center_line]
    ])

    fig, ax = plt.subplots(figsize=(5, 10))

    ax.plot(court_outline[:,0], court_outline[:,1], color='black', linewidth=2)
    ax.scatter(x, y, c='red', s=80)
    ax.plot(Court_center_line[:,0], Court_center_line[:,1], color='black', linewidth=2)

    ax.set_title("Radar")
    ax.set_aspect('equal')
    ax.set_xlim(-Arena_area, Court_Width + Arena_area)
    ax.set_ylim(-Arena_area, Court_Length + Arena_area)
    ax.set_xlabel("Width (meters)")
    ax.set_ylabel("Length (meters)")
    ax.grid(False)

    return fig
