import math

import taichi as ti
import numpy as np

ti.init(arch=ti.gpu)

WIN_WIDTH=800
WIN_HEIGHT=600

gui=ti.GUI("Real Time Raytracer", (WIN_WIDTH, WIN_HEIGHT), fullscreen=True, background_color=0x25A6D9)

pixels=ti.Vector.field(3, dtype=float, shape=(WIN_WIDTH, WIN_HEIGHT))

bounces = 100


class Raytracer:

    def __init__(self):
        self.ch = ComputeHandler()
        self.position = [5.0, 0.0, 0.0]
        self.direction = [-math.sqrt(0.5), math.sqrt(0.5), 0.0]
        self.up = ti.Vector([0.0, 0.0, 1.0])

    def draw(self, iter):
        self.ch.cast_rays(self.position[0], self.position[1], self.position[2], self.direction[0], self.direction[1], self.direction[2])
        self.ch.draw(iter)

    def move(self, move: ti.i32):
        if move == 1:
            self.position = np.add(self.position, [x*0.04 for x in self.direction])
        elif move == 2:
            self.position = np.subtract(self.position, [x*0.04 for x in np.cross(self.direction, self.up)])
        elif move == 3:
            self.position = np.subtract(self.position, [x*0.04 for x in self.direction])
        elif move == 4:
            self.position = np.add(self.position, [x*0.04 for x in np.cross(self.direction, self.up)])

    def add_sphere(self, x, y, z, radius):
        self.ch.add_sphere(x, y, z, radius)


@ti.data_oriented
class ComputeHandler:

    def __init__(self):
        self.rays = ti.Vector.field(n=3, dtype=ti.f32, shape=(WIN_WIDTH, WIN_HEIGHT))
        self.up = ti.Vector([0.0, 0.0, 1.0])

        self.spheres_pos = ti.Vector.field(n=3, dtype=ti.f32, shape=(100, 1))
        self.spheres_radii = ti.field(dtype=ti.f32, shape=(30, 1))
        self.num_spheres = 0

        self.background = ti.Vector([0.5, 0.5, 0.9])

    @ti.kernel
    def draw(self, iterations: ti.f64):
        for i, j in pixels:
            color = pixels[i, j] * ((iterations - 1) / iterations)

            color += self.rays[i, j]/iterations

            pixels[i, j] = color

    @ti.kernel
    def cast_rays(self, p_x: ti.f64, p_y: ti.f64, p_z: ti.f64, d_x: ti.f64, d_y: ti.f64, d_z: ti.f64):
        pos = ti.Vector([p_x, p_y, p_z])
        direction = ti.Vector([d_x, d_y, d_z])
        for i, j in pixels:
            start = pos

            x_off = (i + ti.random(ti.f64) - WIN_WIDTH / 2) / WIN_WIDTH
            y_off = (j + ti.random(ti.f64) - WIN_HEIGHT / 2) / WIN_WIDTH
            ray_dir = direction + x_off * (direction.cross(self.up)) + y_off * self.up

            color = self.background

            for rebound_count in range(bounces):
                hit_object = False
                closestHit = float('inf')

                new_start = ti.Vector([0.0, 0.0, 0.0])
                new_ray_dir = ti.Vector([0.0, 0.0, 0.0])
                new_color = ti.Vector([0.0, 0.0, 0.0])
                for k in range(self.num_spheres):
                    oc = start - self.spheres_pos[k, 0]
                    a = ray_dir.dot(ray_dir)
                    b = 2.0 * oc.dot(ray_dir)
                    c = oc.dot(oc) - self.spheres_radii[k, 0] * self.spheres_radii[k, 0]
                    discriminant = b * b - 4.0 * a * c

                    if discriminant > 0:
                        t = (-b - ti.sqrt(discriminant)) / (2.0 * a)
                        if 0 < t < closestHit:
                            hit_object = True
                            closestHit = t

                            n = (start + ray_dir * t - self.spheres_pos[k, 0]).normalized()
                            new_color = (n * 2.0) + ti.Vector([1.0, 1.0, 1.0])

                            new_start = start + t*ray_dir

                            #new_ray_dir = (start + ray_dir * t) - 2.0*((start + ray_dir * t).dot(n))*n

                            new_ray_dir = (new_start - self.spheres_pos[k, 0]).normalized() + self.random_unit_vector()

                start = new_start
                ray_dir = new_ray_dir
                if hit_object:
                    color = color * 0.5
                else:
                    color = color * 0.5 + self.background * 0.4
                    break

            self.rays[i, j] = color


    @ti.func
    def random_unit_vector(self):
        return ti.Vector([ti.random(ti.f32), ti.random(ti.f32), ti.random(ti.f32)]).normalized()*2 - ti.Vector([1.0, 1.0, 1.0])

    def add_sphere(self, x, y, z, radius):
        self.spheres_pos[self.num_spheres, 1] = ti.Vector([x, y, z])
        self.spheres_radii[self.num_spheres, 1] = radius

        self.num_spheres += 1


if __name__ == '__main__':
    r = Raytracer()

    for a in range(-1, 2):
        for b in range(-1, 2):
            for c in range(-1, 2):
                r.add_sphere(c*1.0, a*1.0, b*1.0, 0.5)
    #r.add_sphere(5.0, 0.0, 0.0, 0.5)
    #r.add_sphere(5.0, 1.1, 0.0, 0.5)
    #r.add_sphere(5.0, -1.1, 0.0, 0.5)

    iterations = 0

    moving_f = False
    moving_b = False
    moving_l = False
    moving_r = False
    while(gui.running):

        iterations+=1.0
        r.draw(iterations)

        for e in gui.get_events(gui.PRESS):
            if e.key==gui.ESCAPE:
                gui.running=False

        gui.get_event()
        if gui.is_pressed('w', ti.GUI.LEFT):
            r.move(1)
            iterations = 0
        if gui.is_pressed('a', ti.GUI.LEFT):
            r.move(2)
            iterations = 0
        if gui.is_pressed('s', ti.GUI.LEFT):
            r.move(3)
            iterations = 0
        if gui.is_pressed('d', ti.GUI.LEFT):
            r.move(4)
            iterations = 0

        gui.set_image(pixels)
        gui.show()

