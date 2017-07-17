import sys
import time

import pygame

from Emotion import *

network = EmotionNetwork(vocab_size, len(emotions), 100, 100, training=False)
network.load("Emotion.npz")

pygame.init()

size = width, height = 1080, 480

screen = pygame.display.set_mode(size)

font = pygame.font.SysFont(None, 64)

text = ""

emotion_smooth = np.zeros((len(emotions)))
emotion_real = np.zeros((len(emotions)))

blink_time = 0

last_time = time.time()

while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()
        if event.type == pygame.KEYDOWN:
            if event.unicode != "":
                if event.key == 8:  # Backspace
                    text = text[:-1]
                elif event.key == 13: # Enter
                    text = ""
                elif event.key == 27:
                    pygame.event.post(pygame.event.Event(pygame.QUIT))
                else:
                    text += event.unicode

                if len(text) > 0:
                    text_as_array = np.zeros((len(text), vocab_size))
                    for i, ch in enumerate(text):
                        if ch in ch_to_ix:
                            text_as_array[i, ch_to_ix[ch]] = 1

                    emotion_real = network.eval([text_as_array])[0]
                else:
                    emotion_real = np.zeros((len(emotions)))

    screen.fill((0, 0, 0))

    guesses = np.argsort(-emotion_smooth)
    y = 64
    for i, guess in enumerate(guesses):
        font_size = int(emotion_smooth[guess] * 128)

        if font_size > 5:
            font_to_use = pygame.font.SysFont(None, font_size)
            em = ix_to_em[guess] + " (%.1f %%)" % (emotion_smooth[guess] * 100)
            text_surf = font_to_use.render(em, True, (255, 255, 0))

            screen.blit(text_surf, (10, y))

            y += font_size

    emotion_smooth = (emotion_smooth * 5 + emotion_real) / 6

    if blink_time > 0.5:
        text_surf = font.render(text + "|", True, (255, 255, 0))
    else:
        text_surf = font.render(text, True, (255, 255, 0))

    screen.blit(text_surf, (0, 0))

    pygame.display.flip()
    
    delta = time.time() - last_time
    blink_time += delta
    if blink_time > 1:
        blink_time -= 1

    last_time = time.time()