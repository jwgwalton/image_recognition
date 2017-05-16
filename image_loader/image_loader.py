from PIL import Image
import numpy as np
import glob


#
# Loads training training_data, each image is returned as a tuple of a pixel array and a vector storing the image type.
# the vector stores a 1 at the index of the classified shape.
#
class ImageLoader(object):

    @staticmethod
    def load_images(path):
        image_list = []
        filenames = glob.glob(path)
        for filename in filenames:
            image = ImageLoader.load_image(filename)
            type_array = np.zeros((3, 1))  #array to store shape in
            type = ImageLoader.parse_file_name(filename)  #enumareted shape type is stored in name
            type_array[type-1] = 1
            image_list.append((image, type_array)) #list of tuples with pixel array and classification
        return image_list

    @staticmethod
    def load_validation_images(path):
        image_list = []
        filenames = glob.glob(path)
        for filename in filenames:
            image = ImageLoader.load_image(filename)
            type = ImageLoader.parse_file_name(filename)  #enumareted shape type is stored in name
            image_list.append((image, type)) #list of tuples with pixel array and classification
        return image_list

    @staticmethod
    def load_image(filename):
        im = Image.open(filename).convert('L')  # returns a numpy array of pixels
        image_array = np.asarray(im)
        image_array = image_array/255
        flat_image_array = np.reshape(image_array, (10000, 1))
        return flat_image_array

    "gross method for stripping shape off filename, relies too much on structure of folders and " \
    "assumption that first character is the shape encoding"
    @staticmethod
    def parse_file_name(file_name):
        file_name_chunks = file_name.split('/')
        data_folder = file_name_chunks[2].split('\\')
        png = data_folder[1]
        return int(png[:1])
