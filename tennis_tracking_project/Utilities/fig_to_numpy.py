import numpy as np

# def fig_to_numpy(fig):
#     fig.canvas.draw()
#     img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
#     img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
#     return img

def fig_to_numpy(fig):
    fig.canvas.draw()

    # Get RGBA buffer
    buf = np.asarray(fig.canvas.buffer_rgba())

    # Convert RGBA → RGB (drop alpha channel)
    img = buf[..., :3]

    return img
