# SPDX-FileCopyrightText: 2021 Andrew Reusch for Adafruit Industries
#
# SPDX-License-Identifier: MIT
import time
import logging
import argparse
import pygame
import os
import sys
import numpy as np
import subprocess

CONFIDENCE_THRESHOLD = 0.5   # at what confidence level do we say we detected a thing
PERSISTANCE_THRESHOLD = 0.25  # what percentage of the time we have to have seen a thing

# App
from rpi_vision.agent.capture import PiCameraStream
from rpi_vision.models.teachablemachine import TeachableMachine

logging.basicConfig()
logging.getLogger().setLevel(logging.INFO)

# initialize the display
pygame.init()
screen = pygame.display.set_mode((0,0), pygame.FULLSCREEN)

capture_manager = PiCameraStream(resolution=(screen.get_width(), screen.get_height()), preview=False)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--include-top', type=bool,
                        dest='include_top', default=True,
                        help='Include fully-connected layer at the top of the network.')

    parser.add_argument('savedmodel', help='TeachableMachine savedmodel')

    parser.add_argument('--tflite',
                        dest='tflite', action='store_true', default=False,
                        help='Convert base model to TFLite FlatBuffer, then load model into TFLite Python Interpreter')
    args = parser.parse_args()
    return args

last_seen = [None] * 10
last_spoken = None

def main(args):
    if len(sys.argv) != 2:
        print("incorrect amount of args")
        sys.exit(1)
    global last_spoken, capture_manager

    # Ask user for input at startup
    user_input = sys.argv[1]
    print("You entered:", user_input)

    capture_manager = PiCameraStream(preview=False)

    if args.rotation in (0, 180):
        buffer = pygame.Surface((screen.get_width(), screen.get_height()))
    else:
        buffer = pygame.Surface((screen.get_height(), screen.get_width()))

    pygame.mouse.set_visible(False)
    screen.fill((0, 0, 0))
    try:
        splash = pygame.image.load(os.path.dirname(sys.argv[0]) + '/bchatsplash.bmp')
        splash = pygame.transform.rotate(splash, args.rotation)
        splash = pygame.transform.scale(splash, (min(screen.get_width(), screen.get_height()), min(screen.get_width(), screen.get_height())))
        screen.blit(splash, ((screen.get_width() - splash.get_width()) // 2, (screen.get_height() - splash.get_height()) // 2))
    except pygame.error:
        pass
    pygame.display.update()

    scale = max(buffer.get_height() // capture_manager.resolution[1], 1)
    scaled_resolution = tuple([x * scale for x in capture_manager.resolution])

    smallfont = pygame.font.Font(None, 24 * scale)
    medfont = pygame.font.Font(None, 36 * scale)
    bigfont = pygame.font.Font(None, 48 * scale)

    model = MobileNetV2Base(include_top=args.include_top)
    capture_manager.start()

    while not capture_manager.stopped:
        if capture_manager.frame is None:
            continue

        buffer.fill((0, 0, 0))
        frame = capture_manager.read()
        previewframe = np.ascontiguousarray(capture_manager.frame)
        img = pygame.image.frombuffer(previewframe, capture_manager.resolution, 'RGB')
        img = pygame.transform.scale(img, scaled_resolution)

        cropped_region = (
            (img.get_width() - buffer.get_width()) // 2,
            (img.get_height() - buffer.get_height()) // 2,
            buffer.get_width(),
            buffer.get_height()
        )
        buffer.blit(img, (0, 0), cropped_region)

        timestamp = time.monotonic()
        prediction = model.tflite_predict(frame)[0] if args.tflite else model.predict(frame)[0]
        delta = time.monotonic() - timestamp
        logging.info(prediction)
        logging.info("%s inference took %d ms, %0.1f FPS" % ("TFLite" if args.tflite else "TF", delta * 1000, 1 / delta))

        input_text_surface = medfont.render(f"Target: {user_input}", True, (0, 200, 255))
        input_text_position = (10, buffer.get_height() - 60)
        buffer.blit(input_text_surface, input_text_position)

        fpstext = "%0.1f FPS" % (1 / delta,)
        fpstext_surface = smallfont.render(fpstext, True, (255, 0, 0))
        buffer.blit(fpstext_surface, fpstext_surface.get_rect(topright=(buffer.get_width() - 10, 10)))

        try:
            temp = int(open("/sys/class/thermal/thermal_zone0/temp").read()) / 1000
            temptext = "%d\N{DEGREE SIGN}C" % temp
            temptext_surface = smallfont.render(temptext, True, (255, 0, 0))
            buffer.blit(temptext_surface, temptext_surface.get_rect(topright=(buffer.get_width() - 10, 30)))
        except OSError:
            pass

        for p in prediction:
            label, name, conf = p
            if conf > CONFIDENCE_THRESHOLD and name.lower() == user_input.lower():
                print("Detected", name)
                persistant_obj = False
                last_seen.append(name)
                last_seen.pop(0)
                inferred_times = last_seen.count(name)
                if inferred_times / len(last_seen) > PERSISTANCE_THRESHOLD:
                    persistant_obj = True

                detecttext = name.replace("_", " ")
                for f in (bigfont, medfont, smallfont):
                    detectsize = f.size(detecttext)
                    if detectsize[0] < screen.get_width():
                        detecttextfont = f
                        break
                else:
                    detecttextfont = smallfont

                detecttext_color = (0, 255, 0) if persistant_obj else (255, 255, 255)
                detecttext_surface = detecttextfont.render(detecttext, True, detecttext_color)
                detecttext_position = (buffer.get_width() // 2, buffer.get_height() - detecttextfont.size(detecttext)[1])
                buffer.blit(detecttext_surface, detecttext_surface.get_rect(center=detecttext_position))

                if persistant_obj and last_spoken != detecttext:
                    subprocess.call(f"echo {detecttext} | festival --tts &", shell=True)
                    last_spoken = detecttext
                break
        else:
            last_seen.append(None)
            last_seen.pop(0)
            if last_seen.count(None) == len(last_seen):
                last_spoken = None

        screen.blit(pygame.transform.rotate(buffer, args.rotation), (0, 0))
        pygame.display.update()

if __name__ == "__main__":
    args = parse_args()
    try:
        main(args)
    except KeyboardInterrupt:
        capture_manager.stop()
