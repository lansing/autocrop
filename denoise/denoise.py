# -*- coding: utf-8 -*-

from __future__ import print_function

import argparse
from random import random

import cv2
import numpy as np
import os
import shutil
import sys

from autocrop.__version__ import __version__

FIXEXP = True  # Flag to fix underexposition
MINFACE = 8  # Minimum face size ratio; too low and we get false positives
INCREMENT = 0.06
GAMMA_THRES = 0.001
GAMMA = 0.90
FACE_RATIO = 6  # Face / padding ratio
QUESTION_OVERWRITE = "Overwrite image files?"
FILETYPES = ['.jpg', '.jpeg', '.bmp', '.dib', '.jp2',
             '.png', '.webp', '.pbm', '.pgm', '.ppm',
             '.sr', '.ras', '.tiff', '.tif']
INPUT_FILETYPES = FILETYPES + [s.upper() for s in FILETYPES]

# Load XML Resource
cascFile = 'haarcascade_frontalface_default.xml'
d = os.path.dirname(sys.modules['autocrop'].__file__)
cascPath = os.path.join(d, cascFile)


# Define simple gamma correction fn
def gamma(img, correction):
    img = cv2.pow(img/255.0, correction)
    return np.uint8(img*255)


def denoise(image, lum):
    denoised = cv2.fastNlMeansDenoisingColored(image, None, lum, lum, 7, 21)

    return denoised


def main(input_d,
         output_d,
         lum):

    reject_count = 0
    output_count = 0
    input_files = [
        os.path.join(input_d, f) for f in os.listdir(input_d)
        if any(f.endswith(t) for t in INPUT_FILETYPES)
    ]
    if output_d is None:
        output_d = input_d

    # Guard against calling the function directly
    input_count = len(input_files)
    assert input_count > 0

    for input_filename in input_files:
        basename = os.path.basename(input_filename)
        output_filename = os.path.join(output_d, basename)

        # Attempt the crop
        input_img = cv2.imread(input_filename)

        if isinstance(input_img, type(None)):
            print('Skipping: {}'.format(input_filename))
            reject_count += 1
        else:
            image = denoise(input_img, lum)

            if isinstance(image, type(None)):
                print('Skipping: {}'.format(input_filename))
                reject_count += 1
            else:
                cv2.imwrite(output_filename, image)
                print('Denoising:    {}'.format(output_filename))
                output_count += 1

    # Stop and print status
    print('{} input files, {} images denoised, {} rejected'.format(
        input_count, output_count, reject_count))


def input_path(p):
    """Returns path, only if input is a valid directory"""
    no_folder = 'Input folder does not exist'
    no_images = 'Input folder does not contain any image files'
    p = os.path.abspath(p)
    if not os.path.isdir(p):
        raise argparse.ArgumentTypeError(no_folder)
    filetypes = set(os.path.splitext(f)[-1] for f in os.listdir(p))
    if not any(t in INPUT_FILETYPES for t in filetypes):
        raise argparse.ArgumentTypeError(no_images)
    else:
        return p


def output_path(p):
    """Returns path, if input is a valid directory name.
    If directory doesn't exist, creates it."""
    p = os.path.abspath(p)
    if not os.path.isdir(p):
        os.makedirs(p)
    return p


def size(i):
    """Returns valid only if input is a positive integer under 1e5"""
    error = 'Invalid pixel size'
    try:
        i = int(i)
    except TypeError:
        raise argparse.ArgumentTypeError(error)
    if i > 0 and i < 1e5:
        return i
    else:
        raise argparse.ArgumentTypeError(error)


def compat_input(s=''):
    """Compatibility function to permit testing for Python 2 and 3"""
    try:
        return raw_input(s)
    except NameError:
        return input(s)


def confirmation(question, default=True):
    """Ask a yes/no question via standard input and return the answer.

    If invalid input is given, the user will be asked until
    they acutally give valid input.

    Args:
        question(str):
            A question that is presented to the user.
        default(bool|None):
            The default value when enter is pressed with no value.
            When None, there is no default value and the query
            will loop.
    Returns:
        A bool indicating whether user has entered yes or no.

    Side Effects:
        Blocks program execution until valid input(y/n) is given.
    """
    yes_list = ["yes", "y"]
    no_list = ["no", "n"]

    default_dict = {  # default => prompt default string
        None: "[y/n]",
        True: "[Y]/n",
        False: "y/[N]",
    }

    default_str = default_dict[default]
    prompt_str = "%s %s " % (question, default_str)

    while True:
        choice = compat_input(prompt_str).lower()

        if not choice and default is not None:
            return default
        if choice in yes_list:
            return True
        if choice in no_list:
            return False

        notification_str = "Please respond with 'y' or 'n'"
        print(notification_str)


def parse_args(args):
    help_d = {
        'desc': 'Denoises images',
        'input': '''Folder where images to denoise are located. Default:
                     current working directory''',
        'output': '''Folder where denoised images will be moved to.''',
        'lum': '''Big lum value perfectly removes noise but also removes image details,
                smaller lum value preserves details but also preserves some noise'''

    }

    parser = argparse.ArgumentParser(description=help_d['desc'])
    parser.add_argument('-i', '--input', default='.', type=input_path,
                        help=help_d['input'])
    parser.add_argument('-o', '--output', '-p', '--path', type=output_path, help=help_d['output'])
    parser.add_argument('-l', '--lum', type=float, help=help_d['lum'])
    return parser.parse_args()


def cli():
    args = parse_args(sys.argv[1:])
    if args.input == args.output:
        args.output = None
    print('Processing images in folder:', args.input)
    main(args.input,
         args.output,
         args.lum)
