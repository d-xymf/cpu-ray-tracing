from PIL import Image
import numpy as np
import HitInfo
import Sphere


# the render result will be saved here
image = Image.new(mode="RGB", size=(400, 300))
width, height = image.size
aspect_ratio = width/height

# define a sphere at (0,0,4) with radius 1 and color (1.0,0.2,0.4)
sphere = Sphere.Sphere(np.array([0.0,0.0,4.0]), 1.0, np.array([1.0,0.2,0.4]))

# define camera with vfov 60 degrees
# always pointing in +z direction
cam_vfov = 40.0 * 3.14159265/180.0
near = 1.0/np.tan(cam_vfov)
# define a point light source at (-1, 3, 3)
light_pos = np.array([-2.0,3.0,1.0])

# c - center of the sphere
# r - radius of the sphere
# rd - ray direction
def ray_sphere_intersection(sc: np.array, sr: float, rd: np.array) -> HitInfo.HitInfo:

    # coefficients for quadratic equation
    a = np.dot(rd, rd)
    b = -2*np.dot(rd, sc)
    c = np.dot(sc, sc) - sr*sr

    # discard if sphere is behind us or we are inside the sphere
    if b > 0 or c < 0:
        return HitInfo.HitInfo(False, None, None)
    
    # check for hit
    d = b*b - 4*a*c # discriminant
    if d < 0:
        return HitInfo.HitInfo(False, None, None)# no hit
    else:
        k = (-b - np.sqrt(d)) / (2*a) # this scalar multiplied by rd gives coords of intersection

        intersection = rd * k

        normal = (k*rd - sc) / np.linalg.norm(k*rd - sc)

        return HitInfo.HitInfo(True, intersection, normal)

# fragment shader
def frag(frag_coord: tuple[int, int]) -> tuple[int, int, int]:

    # 0.0 - 1.0
    uv = np.array(frag_coord) / np.array(image.size)
    # -1.0 - 1.0
    uv = uv * 2.0 - np.array([1.0, 1.0])
    uv[0] = uv[0] * aspect_ratio

    # create a ray for this pixel
    ray_dir = np.append(uv, near)
    # ray direction must be normalized 
    # for sphere intersection calculation to work
    ray_dir = ray_dir / np.linalg.norm(ray_dir)

    # if ray intersects sphere make pixel white else black
    color = np.array([0.0, 0.0, 0.0])
    ray_hit_info = ray_sphere_intersection(sphere.center, sphere.radius, ray_dir)
    if ray_hit_info.hit:

        light_dir = light_pos - ray_hit_info.coords
        light_dir = light_dir / np.linalg.norm(light_dir)

        normal = ray_hit_info.normal

        # simple diffuse lighting
        color = sphere.color * np.dot(light_dir, normal)

    else:
        color = np.array([0.17, 0.2, 0.23])

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
