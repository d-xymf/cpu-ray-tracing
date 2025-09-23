from PIL import Image
import numpy as np

# the render result will be saved here
image = Image.new(mode="RGB", size=(800, 600))

# fragment shader
def frag(frag_coord: tuple[int, int]) -> tuple[int, int, int]:

    uv = np.array(frag_coord) / np.array(image.size)

    color = np.append(np.array(uv), 0.0)

    # convert np array to tuple of ints (0.0-1.0 -> 0 - 255)
    int_color = np.rint(color * 255).astype(int)

    return tuple(int_color)

# actual rendering of the image
width, height = image.size
for x in range(width):
    for y in range(height):

        color = frag((x, y))

        # 0,1----------1,1
        #  |            |
        #  |            |
        # 0,0----------1,0
        image.putpixel((x, height-1-y), color)

image.save("output.png")
