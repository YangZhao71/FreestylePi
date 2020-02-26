#!/usr/bin/env python
# encoding: utf-8

import os
import time
import math
import random
import pygame

# Color constants

COLOR_BLACK = (0, 0, 0)
COLOR_WHITE = (255, 255, 255)
COLOR_RED = (255, 0, 0)
COLOR_GREEN = (0, 255, 0)
COLOR_BLUE = (0, 0, 255)


class TFTDisplay:

    def __init__(self):
        self.screen = self.init_pygame()
        self.font = pygame.font.Font("data/DejaVuSansMono.ttf", 22)
        self.pos = (160, 120)
        self.start_time = time.time()

    def display_text(self, text="",
                     color=COLOR_WHITE,
                     clear=True):
        if clear:
            self.reset()
        self.surface = self.font.render(text, True, color)
        self.rect = self.surface.get_rect(center=self.pos)
        self.screen.blit(self.surface, self.rect)
        pygame.display.update()

    def display_image():
        pass

    def display_wave_block(self):
        import math, time, random
        surface = pygame.Surface((320,240))
        surface.fill(COLOR_BLACK)

        start_time = time.time()
        frequency = 4
        amplitude = 50

        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_c and pygame.key.get_mods() & pygame.KMOD_CTRL:
                        print("exit...")
                        sys.exit()
            surface.fill(COLOR_BLACK)
            speed = 1
            for x in range(0, 320):
                if time.time() - start_time > 0.1:
                    frequency = 4
                    start_time = time.time()
                    amplitude = 50 * random.random()

                y = int(120 + amplitude*math.sin(frequency*((float(x)/320)*(2*math.pi) + (speed*time.time()))))
                surface.set_at((x, y), COLOR_BLUE)
            self.screen.blit(surface, (0,0))
            pygame.display.update()

    def display_wave(self, background=COLOR_BLACK):
        import math, time, random
        surface = pygame.Surface((320,240))
        surface.fill(background)

        frequency = 4
        amplitude = 40
        speed = 1

        for x in range(0, 320):
            y = int(100 + amplitude*math.sin(frequency*((float(x)/320)*(2*math.pi) + (speed*time.time()))))
            surface.set_at((x, y), COLOR_BLUE)
        self.screen.blit(surface, (0,0))
        pygame.display.update()


    def reset(self):
        self.screen.fill(COLOR_BLACK)
        pygame.display.update()

    def init_pygame(self, mouse=False,
                    display_size=(320, 240),
                    background=COLOR_BLACK):
        '''Init pygame
        '''
        pygame.init()
        pygame.mouse.set_visible(mouse)
        screen = pygame.display.set_mode(display_size)
        screen.fill(background)
        pygame.display.update()
        return screen

    def init_env(self):
        """Set env variables
        """
        os.putenv('SDL_VIDEODRIVER', 'fbcon')
        os.putenv('SDL_FBDEV', '/dev/fb1')
        os.putenv('SDL_MOUSEDRV', 'TSLIB')
        os.putenv('SDL_MOUSEDEV', '/dev/input/touchscreen')

    def display_curves(self, background=COLOR_BLACK):
        def tanh(x):
            return (1.0 - math.exp(-x)) / (1.0 + math.exp(-x))

        def sech(x):
            return 2.0 / (math.exp(-x) + math.exp(x))

        def t1(x):
            return sech(x) * sech(x)

        def t2(x):
            return -2.0 * tanh(x) * sech(x) * sech(x)

        def t3(x):
            return 4.0 * math.pow(tanh(x)*sech(x), 2) - 2.0 * math.pow(sech(x), 4)

        def t4(x):
            return 16.0 * tanh(x) * math.pow(sech(x), 4) - 8.0 * pow(tanh(x), 3) * pow(sech(x), 2)

        def t5(x):
            return 8.0 * math.pow(sech(x), 2) * \
                (2.0 * math.pow(tanh(x), 4) + 2.0 * math.pow(sech(x), 4) - \
                    11.0 * math.pow(tanh(x), 2) * math.pow(sech(x), 2))

        def t6(x):
            return -16.0 * tanh(x) * math.pow(sech(x), 2) * \
                (2.0 * math.pow(tanh(x), 4) + 17 * math.pow(sech(x), 4) -
                    26 * math.pow(tanh(x)*sech(x), 2))

        surface = pygame.Surface((320,240))
        surface.fill(background)

        amplitude = 20
        shift = 120

        for x in range(0, 320):
            xt = ((float(x)/320) - 0.5)*6

            y1 = int(shift + amplitude * t1(xt))
            surface.set_at((x, y1), (102, 51, 0))

            y2 = int(shift + amplitude * t2(xt))
            surface.set_at((x, y2), (204, 0, 0))

            y3 = int(shift + amplitude * t3(xt))
            surface.set_at((x, y3), (255, 255, 0))

            y4 = int(shift + amplitude * t4(xt))
            surface.set_at((x, y4), (127, 0, 255))
            
            y5 = int(shift + amplitude * t5(xt) * 0.2)
            surface.set_at((x, y5), (0, 0, 255))

            y6 = int(shift + amplitude * t6(xt) * 0.1)
            surface.set_at((x, y6), (255, 0, 0))

        self.screen.blit(surface, (0, 0))
        pygame.display.update()


if __name__ == "__main__":
    tft = TFTDisplay()
    tft.display_curves()

