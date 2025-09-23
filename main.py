from PIL import Image
import numpy as np

# the render result will be saved here
image = Image.new(mode="RGB", size=(400, 300))
width, height = image.size
aspect_ratio = width/height

# define a sphere at (0,0,0) with radius 1
sphere_center = np.array([0.0,0.0,4.0])
sphere_radius = 1.0

# define camera with vfov 60 degrees
# always pointing in +z direction
cam_vfov = 40.0 * 3.14159265/180.0
near = 1.0/np.tan(cam_vfov)

# c - center of the sphere
# r - radius of the sphere
# rd - ray direction
def ray_sphere_intersection(c, r, rd):

    k = np.dot(c, rd)
    # discard if sphere is behind us
    if k < 0:
        return False

    if np.linalg.norm(k*rd - c) <= r:
        return True
    else:
        return False

# fragment shader
def frag(frag_coord: tuple[int, int]) -> tuple[int, int, int]:

    # 0.0 - 1.0
    uv = np.array(frag_coord) / np.array(image.size)
    # -1.0 - 1.0
    uv = uv * 2.0 - np.array([1.0, 1.0])
    uv[0] = uv[0] * aspect_ratio

    # create a ray for this pixel
    ray_dir = np.append(uv, near)
    ray_dir = ray_dir / np.linalg.norm(ray_dir)

    # if ray intersects sphere make pixel white else black
    color = np.array([0.0, 0.0, 0.0])
    if ray_sphere_intersection(sphere_center, sphere_radius, ray_dir):
        color = np.array([1.0, 1.0, 1.0])

    # convert np array to tuple of ints (0.0-1.0 -> 0-255)
    int_color = np.rint(color * 255).astype(int)

    return tuple(int_color)

# actual rendering of the image
print("started rendering")
for x in range(width):
    for y in range(height):

        color = frag((x, y))

        # 0,1----------1,1
        #  |            |
        #  |            |
        # 0,0----------1,0
        image.putpixel((x, height-1-y), color)

print("finished rendering")

image.save("output.png")
