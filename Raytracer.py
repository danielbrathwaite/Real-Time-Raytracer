import math

import taichi as ti
import numpy as np

ti.init(arch=ti.gpu)

WIN_WIDTH=800
WIN_HEIGHT=600

gui=ti.GUI("Real Time Raytracer", (WIN_WIDTH, WIN_HEIGHT), fullscreen=True, background_color=0x25A6D9)

pixels=ti.Vector.field(3, dtype=float, shape=(WIN_WIDTH, WIN_HEIGHT))
drawpixels=ti.Vector.field(3, dtype=float, shape=(WIN_WIDTH, WIN_HEIGHT))

bounces = 5
small = 0.0001
scale = 1


class Raytracer:

    def __init__(self):
        self.ch = ComputeHandler()
        self.position = [5.0, 0.0, 0.0]
        self.direction = [-1.0, 0.0, 0.0]
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
        elif move == 5:
            self.position = np.add(self.position, [x*0.04 for x in self.up])
        elif move == 6:
            self.position = np.subtract(self.position, [x*0.04 for x in self.up])

    def add_sphere(self, x, y, z, radius):
        self.ch.add_sphere(x, y, z, radius)

    def add_light(self, x, y, z, radius):
        self.ch.add_light(x, y, z, radius)


@ti.data_oriented
class ComputeHandler:

    def __init__(self):
        self.rays = ti.Vector.field(n=3, dtype=ti.f32, shape=(WIN_WIDTH, WIN_HEIGHT))
        self.up = ti.Vector([0.0, 0.0, 1.0])

        self.spheres_pos = ti.Vector.field(n=3, dtype=ti.f32, shape=(30, 1))
        self.spheres_radii = ti.field(dtype=ti.f32, shape=(30, 1))
        self.num_spheres = 0

        self.lights_pos = ti.Vector.field(n=3, dtype=ti.f32, shape=(100, 1))
        self.light_radii = ti.field(dtype=ti.f32, shape=(30, 1))
        self.num_lights = 0

        self.background = ti.Vector([0.5, 0.5, 0.9])

    @ti.kernel
    def draw(self, iterations: ti.f64):
        for i, j in pixels:
            color = pixels[i, j] * ((iterations - 1) / iterations)

            color += self.rays[i, j]/iterations

            pixels[i, j] = color
        for i, j in drawpixels:
            drawpixels[i, j] = pixels[i//scale*scale, j//scale*scale]


    @ti.kernel
    def cast_rays(self, p_x: ti.f64, p_y: ti.f64, p_z: ti.f64, d_x: ti.f64, d_y: ti.f64, d_z: ti.f64):
        pos = ti.Vector([p_x, p_y, p_z])
        direction = ti.Vector([d_x, d_y, d_z])
        for i, j in pixels:
            start = pos

            alpha = 0.2

            x_off = (i + ti.random(ti.f64) - WIN_WIDTH / 2) / WIN_WIDTH
            y_off = (j + ti.random(ti.f64) - WIN_HEIGHT / 2) / WIN_WIDTH
            ray_dir = direction + x_off * (direction.cross(self.up)) + y_off * self.up

            cur_pixel = ti.Vector([0.0, 0.0, 0.0])

            for rebound_count in range(1):
                hit_object = False
                closestHit = float('inf')
                hit_sphere = -1
                intersection = ti.Vector([0.0, 0.0, 0.0])

                new_start = ti.Vector([0.0, 0.0, 0.0])
                new_ray_dir = ti.Vector([0.0, 0.0, 0.0])
                for k in range(self.num_spheres):
                    oc = start - self.spheres_pos[k, 0]
                    a = ray_dir.dot(ray_dir)
                    b = 2.0 * oc.dot(ray_dir)
                    c = oc.dot(oc) - self.spheres_radii[k, 0] * self.spheres_radii[k, 0]
                    discriminant = b * b - 4.0 * a * c

                    if discriminant > 0:
                        t = (-b - ti.sqrt(discriminant)) / (2.0 * a)
                        n = (start + ray_dir * t - self.spheres_pos[k, 0]).normalized()

                        if 0 < t < closestHit and ray_dir.dot(n) < 0:
                            hit_object = True
                            closestHit = t
                            intersection = start+ray_dir*t+n*0.001

                            new_start = start + t*ray_dir
                            #new_ray_dir = (start + ray_dir * t) - 2.0*((start + ray_dir * t).dot(n))*n
                            cur_pixel = n/2 + ti.Vector([0.5,0.5,0.5])

                            new_ray_dir = (new_start - self.spheres_pos[k, 0]).normalized() + self.random_unit_vector()

                if hit_object:
                    color = ti.Vector([0.6, 0.5, 0.5]) * self.light_sample(intersection[0], intersection[1], intersection[2])
                    #cur_pixel = (1 - alpha) * color + alpha * cur_pixel
                else:
                    break

                start = new_start
                ray_dir = new_ray_dir

            self.rays[i, j] = cur_pixel


    @ti.func
    def random_unit_vector(self):
        return ti.Vector([ti.random(ti.f32), ti.random(ti.f32), ti.random(ti.f32)]).normalized()*2.0 - ti.Vector([1.0, 1.0, 1.0])

    @ti.func
    def light_sample(self, p_x: ti.f64, p_y: ti.f64, p_z: ti.f64) -> ti.f64:
        start = ti.Vector([p_x, p_y, p_z])
        light = 1.0
        for i in range(self.num_lights):
            ray_dir = self.lights_pos[i, 0] - start + self.random_unit_vector() * self.light_radii[i, 0]
            for k in range(self.num_spheres):
                oc = start - self.spheres_pos[k, 0]
                a = ray_dir.dot(ray_dir)
                b = 2.0 * oc.dot(ray_dir)
                c = oc.dot(oc) - self.spheres_radii[k, 0] * self.spheres_radii[k, 0]
                discriminant = b * b - 4.0 * a * c

                if discriminant > 0:
                    t = (-b - ti.sqrt(discriminant)) / (2.0 * a)
                    n = (start + ray_dir * t - self.spheres_pos[k, 0]).normalized()
                    if 0 < t:
                        light = 0.0
        return light

    def add_sphere(self, x, y, z, radius):
        self.spheres_pos[self.num_spheres, 1] = ti.Vector([x, y, z])
        self.spheres_radii[self.num_spheres, 1] = radius

        self.num_spheres += 1

    def add_light(self, x, y, z, radius):
        self.lights_pos[self.num_lights, 1] = ti.Vector([x, y, z])
        self.light_radii[self.num_lights, 1] = radius

        self.num_lights += 1


if __name__ == '__main__':
    r = Raytracer()

    for a in range(-1, 2):
        for b in range(-1, 2):
            for c in range(-1, 2):
                r.add_sphere(c*1.01, a*1.01, b*1.01, 0.5)
    r.add_light(10.0, 10.0, 10.0, 1.0)
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
        if gui.is_pressed(gui.SPACE, ti.GUI.LEFT):
            r.move(5)
            iterations = 0
        if gui.is_pressed(gui.SHIFT, ti.GUI.LEFT):
            r.move(6)
            iterations = 0

        gui.set_image(drawpixels)
        gui.show()

