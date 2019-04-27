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


def crop_positions(imgh, imgw, x, y, w, h,
                   fheight, fwidth, facePercent,
                   padUp, padDown, padLeft, padRight):
    # Check padding values
    padUp = 50 if (padUp is False or padUp < 0) else padUp
    padDown = 50 if (padDown is False or padDown < 0) else padDown
    padLeft = 50 if (padLeft is False or padLeft < 0) else padLeft
    padRight = 50 if (padRight is False or padRight < 0) else padRight

    # enfoce face percent
    facePercent = 100 if facePercent > 100 else facePercent
    facePercent = 50 if facePercent <= 0 else facePercent

    # Adjust output height based on Face percent
    height_crop = h * 100.0 / facePercent

    # Ensure height is within boundaries
    height_crop = imgh if height_crop > imgh else height_crop

    aspect_ratio = float(fwidth) / float(fheight)
    # Calculate width based on aspect ratio
    width_crop = aspect_ratio * float(height_crop)

    # Calculate padding by centering face
    xpad = (width_crop - w) / 2
    ypad = (height_crop - h) / 2

    # Calc. positions of crop
    h1 = float(x - (xpad * padLeft / (padLeft + padRight)))
    h2 = float(x + w + (xpad * padRight / (padLeft + padRight)))
    v1 = float(y - (ypad * padUp / (padUp + padDown)))
    v2 = float(y + h + (ypad * padDown / (padUp + padDown)))

    # Move crop inside photo boundaries
    while h1 < 0:
        h1 = h1 + 1
        h2 = h2 + 1
    while v1 < 0:
        v1 = v1 + 1
        v2 = v2 + 1
    while v2 > imgh:
        v2 = v2 - 1
        h2 = h2 - 1 * aspect_ratio
    while h2 > imgw:
        h2 = h2 - 1
        v2 = v2 - 1 / aspect_ratio

    return [int(v1), int(v2), int(h1), int(h2)]


def crop(image, fheight=500, fwidth=500,
         outerPad = 0,
         border='replicate'):
    """Given a ndarray image with a face, returns cropped array.

    Arguments:
        - image, the numpy array of the image to be processed.
        - fwidth, the final width (px) of the cropped img. Default: 500
        - fheight, the final height (px) of the cropped img. Default: 500
        - padUp, Padding from top
        - padDown, Padding to bottom
        - padLeft, Padding from left
        - padRight, Padding to right
    Returns:
        - image, a cropped numpy array

    ndarray, int, int -> ndarray
    """
    # Some grayscale color profiles can throw errors, catch them
    try:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    except cv2.error:
        gray = image

    # Scale the image
    height, width = (image.shape[:2])

    if height > width:
        uheight = fheight
        uwidth = int(width * fheight / height)
    else:
        uheight = int(height * fwidth / width)
        uwidth = fwidth

    uwidth = uwidth - outerPad
    uheight = uheight - outerPad

    image = cv2.resize(image, (uwidth, uheight),
                       interpolation=cv2.INTER_AREA)

    y_pad = max([int((fheight - uheight) / 2), 0])
    x_pad = max([int((fwidth - uwidth) / 2), 0])

    if border == 'replicate' or border == 'reflect':
        if border == 'replicate':
            border_type = cv2.BORDER_REPLICATE
        elif border == 'reflect':
            border_type = cv2.BORDER_REFLECT_101

        image = cv2.copyMakeBorder(image, y_pad, y_pad, x_pad, x_pad, border_type)

    elif border == 'average':
        corner_fill_zone = image[2:6, 2:6, ]
        fill_color = np.mean(corner_fill_zone, (0, 1))

        image = cv2.copyMakeBorder(image, y_pad, y_pad, x_pad, x_pad, cv2.BORDER_CONSTANT, None, fill_color)

        if image.shape[:2] != (fheight, fwidth):
            y_extra_pad = fheight - image.shape[0]
            x_extra_pad = fwidth - image.shape[1]
            image = cv2.copyMakeBorder(image, y_extra_pad, 0, x_extra_pad, 0, cv2.BORDER_CONSTANT, None, fill_color)

    elif border == 'inpaint':
        mask = np.ones((fwidth, fheight)).astype('uint8') * 1
        mask[y_pad+1:(y_pad + uheight), x_pad+1:(x_pad + uwidth)] = 0

        corner_fill_zone = image[2:6, 2:6, ]
        fill_color = np.mean(corner_fill_zone, (0, 1))

        image = cv2.copyMakeBorder(image, y_pad, y_pad, x_pad, x_pad, cv2.BORDER_CONSTANT, None, fill_color)

        if image.shape[:2] != (fheight, fwidth):
            y_extra_pad = fheight - image.shape[0]
            x_extra_pad = fwidth - image.shape[1]
            image = cv2.copyMakeBorder(image, 0, y_extra_pad, 0, x_extra_pad, cv2.BORDER_CONSTANT, None, fill_color)

        image_inpaint = cv2.inpaint(image, mask, 8, cv2.INPAINT_NS)

        image = cv2.addWeighted(image, 0.0, image_inpaint, 1.0, 0)

    else:
        raise Exception("unsupported border, use replicate or reflect")

    return image


def main(input_d,
         output_d,
         reject_d,
         fheight=500,
         fwidth=500,
         outerPad = 0,
         border='replicate'):
    """Crops folder of images to the desired height and width if a face is found

    If input_d == output_d or output_d is None, overwrites all files
    where the biggest face was found.

    Args:
        input_d (str): Directory to crop images from.
        output_d (str): Directory where cropped images are placed.
        reject_d (str): Directory where images that cannot be cropped are
                        placed.
        fheight (int): Height (px) to which to crop the image.
                       Default: 500px
        fwidth (int): Width (px) to which to crop the image.
                       Default: 500px
    Side Effects:
        Creates image files in output directory.

    str, str, (int), (int) -> None
    """
    reject_count = 0
    output_count = 0
    input_files = [
        os.path.join(input_d, f) for f in os.listdir(input_d)
        if any(f.endswith(t) for t in INPUT_FILETYPES)
    ]
    if output_d is None:
        output_d = input_d
    if reject_d is None and output_d is None:
        reject_d = input_d
    if reject_d is None:
        reject_d = output_d

    # Guard against calling the function directly
    input_count = len(input_files)
    assert input_count > 0

    for input_filename in input_files:
        basename = os.path.basename(input_filename)
        output_filename = os.path.join(output_d, basename)
        reject_filename = os.path.join(reject_d, basename)

        # Attempt the crop
        input_img = cv2.imread(input_filename)

        if isinstance(input_img, type(None)):
            print('Skipping: {}'.format(reject_filename))
            reject_count += 1
        else:
            image = crop(input_img,
                         fheight,
                         fwidth,
                         outerPad,
                         border)

            # Did the crop produce a valid image
            if isinstance(image, type(None)):
                # if input_filename != reject_filename:
                #     # Move the file to the reject directory
                #     shutil.move(input_filename, reject_filename)
                print('Skipping: {}'.format(reject_filename))
                reject_count += 1
            else:
                # if input_filename != output_filename:
                #     Move the file to the output directory
                #     shutil.move(input_filename, output_filename)
                # Write the cropped image
                cv2.imwrite(output_filename, image)
                print('Cropping:    {}'.format(output_filename))
                output_count += 1

    # Stop and print status
    print('{} input files, {} faces cropped, {} rejected'.format(
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
            'desc': 'Automatically crops faces from batches of pictures',
            'input': '''Folder where images to crop are located. Default:
                     current working directory''',
            'output': '''Folder where cropped images will be moved to.

                      Default: current working directory, meaning images are
                      cropped in place.''',
            'reject': '''Folder where images that could not be cropped will be
                       moved to.

                      Default: current working directory, meaning images that
                      are not cropped will be left in place.''',
            'width': 'Width of cropped files in px. Default=500',
            'height': 'Height of cropped files in px. Default=500',
            'y': 'Bypass any confirmation prompts',
            'facePercent': 'Percentage of face to image height',
            'outerPad': 'Padding to add to every image',
            'border': '''type of border
                        default: replicate''',
            'facepad': '''proportion to add around face 
                            default: 0.0'''
            }

    parser = argparse.ArgumentParser(description=help_d['desc'])
    parser.add_argument('-i', '--input', default='.', type=input_path,
                        help=help_d['input'])
    parser.add_argument('-o', '--output', '-p', '--path', type=output_path,
                        default=None, help=help_d['output'])
    parser.add_argument('-r', '--reject', type=output_path, default=None,
                        help=help_d['reject'])
    parser.add_argument('-w', '--width', type=size,
                        default=500, help=help_d['width'])
    parser.add_argument('-H', '--height',
                        type=size, default=500, help=help_d['height'])
    parser.add_argument('-v', '--version', action='version',
                        version='%(prog)s version {}'.format(__version__))
    parser.add_argument('--no-confirm', action='store_true', help=help_d['y'])
    parser.add_argument('--outerPad', type=size,
                        default=0, help=help_d['outerPad'])
    parser.add_argument('--border', type=str, default='replicate', help=help_d['border'])
    return parser.parse_args()


def cli():
    args = parse_args(sys.argv[1:])
    if not args.no_confirm:
        if args.output is None or args.input == args.output:
            if not confirmation(QUESTION_OVERWRITE):
                sys.exit()
    if args.input == args.output:
        args.output = None
    print('Processing images in folder:', args.input)
    main(args.input,
         args.output,
         args.reject,
         args.height,
         args.width,
         args.outerPad,
         args.border)
