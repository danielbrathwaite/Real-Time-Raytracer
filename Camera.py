import taichi as ti


@ti.data_oriented
class Camera:

    def __init__(self, pixelwidth, pixelheight, fov):
        self.iterations = 0
        self.pixelwidth=pixelwidth
        self.pixelheight=pixelheight
        self.screen_dist = min(pixelwidth, pixelheight)/fov
        self.position = ti.Vector([0.0, 0.0, 0.0])
        self.direction = ti.Vector([self.screen_dist, 0.0, 0.0])
        self.right = ti.vector([])

    @ti.kernel
    def draw(self, pixels):
        self.iterations +=1

        for i, j in pixels:
            color = pixels[i, j] * ((self.iterations-1) /self.iterations)

            color += self.cast_ray(i, j) / self.iterations

    def cast_ray(self, i, j):
        x=1

    def move(self, vel):
        self.iterations =0
        self.position +=vel