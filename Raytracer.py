

import taichi as ti
import Camera

ti.init(arch=ti.gpu)

WIN_WIDTH=800
WIN_HEIGHT=600

gui=ti.GUI("Real Time Raytracer", (WIN_WIDTH, WIN_HEIGHT), fullscreen=True, background_color=0x25A6D9)

pixels=ti.Vector.field(3, dtype=float, shape=(WIN_WIDTH, WIN_HEIGHT))


class Raytracer:

    def __init__(self):
        self.camera=Camera()
        self.camera_move_speed=0.03


    def draw(self, camera):
        self.camera.draw()





if __name__ == '__main__':
    
    while(gui.running):

        for e in gui.get_events(gui.PRESS):
            if e.key==gui.ESCAPE:
                gui.running=False

        gui.set_image(pixels)
        gui.show()

