import os
import random
from scipy import ndarray

import skimage as sk
from skimage import transform
from skimage import util
from skimage import io

def random_rotation(image_array: ndarray):
    # pick a random degree of rotation between 25% on the left and 25% on the right
    random_degree = random.uniform(-25, 25)
    return sk.transform.rotate(image_array, random_degree)

def random_noise(image_array: ndarray):
    # add random noise to the image
    return sk.util.random_noise(image_array)

def horizontal_flip(image_array: ndarray):
    return image_array[:, ::-1]

# dictionary of the transformations
available_transformations = {
    'rotate': random_rotation,
    'noise': random_noise,
    'horizontal_flip': horizontal_flip
}

# get requested number of images from largest folder
master_path = r'__PATH_TO_LARGEST_FOLDER_IN_TRAIN_DATASET__'
target_num = len([f for f in os.listdir(master_path)])
print('All categories must be expanded up to', target_num, 'files')

paths = [r'__PATHS_TO_OTHER_FOLDERS_IN_TRAIN_DATASET__']
for path in paths:
    images = [f for f in os.listdir(path)]
    initial_num = len(images)
    idx = 0
    for i in range(target_num - initial_num):
        image_path = os.path.join(path, images[idx])
        if idx+1 == len(images):
            idx = 0
        else:
            idx += 1

        image_to_transform = sk.io.imread(image_path)
        num_transformations_to_apply = random.randint(1, len(available_transformations)*2)

        num_transformations = 0
        transformed_image = None
        for num_transformations in range(num_transformations_to_apply):
            key = random.choice(list(available_transformations))
            transformed_image = available_transformations[key](image_to_transform)
            num_transformations += 1

        new_file_path = os.path.join(path, f'augmented_{i+1}_{images[idx]}')

        io.imsave(new_file_path, transformed_image)