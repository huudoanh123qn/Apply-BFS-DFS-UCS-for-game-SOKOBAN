import pygame
from pygame.locals import *
import constants as SOKOBAN
from game import *


def main():
    pygame.init()
    pygame.key.set_repeat(100, 100)
    #window = pygame.display.set_mode((SOKOBAN.MAPWIDTH * SOKOBAN.SPRITESIZE, SOKOBAN.MAPHEIGHT * SOKOBAN.SPRITESIZE))
    window = pygame.display.set_mode((SOKOBAN.WINDOW_WIDTH, SOKOBAN.WINDOW_HEIGHT))

    run = True
    while run:
        event = pygame.event.wait()
        if event.type == QUIT:
            run = False
        if event.type == KEYDOWN:
            if event.key == K_j:
                sokoban = Game(window)
                sokoban.start()
            elif event.key == K_ESCAPE:
                run = False

        # pygame.draw.rect(window, (0,255,0), (0,0,SOKOBAN.MAPWIDTH * SOKOBAN.SPRITESIZE,SOKOBAN.MAPHEIGHT * SOKOBAN.SPRITESIZE))
        pygame.draw.rect(window, SOKOBAN.BLUE, (0,0,SOKOBAN.WINDOW_WIDTH,SOKOBAN.WINDOW_HEIGHT))
        pygame.display.flip()

    pygame.quit()


if __name__ == "__main__":
    main()