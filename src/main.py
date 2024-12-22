
from utils import *
from classification.classifier import *
from postprocessing.notes import *
from segmentation.segmentor import *
from preprocessing.staff import *
from preprocessing import DeskewProcessor, BinarizationProcessor, NoiseRemovalProcessor
from typing import Optional
from skimage.draw import rectangle
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, recall_score, classification_report, accuracy_score
import torch.nn.functional as F
from numpy.linalg import norm
import os
import joblib
import traceback
import pickle
import random
import os 


class ImageProcessingPipeline:
    """
    Pipeline to process an image through multiple processors.
    """
    def __init__(self, binarization_method: str = 'otsu', deskew_processor: Optional[DeskewProcessor] = None):
        """
        Initialize the ImageProcessingPipeline.

        Args:
            binarization_method (str): Binarization method. Default is 'otsu'.
            deskew_processor (Optional[DeskewProcessor]): Deskew processor.
        """
        self.binarization_method = binarization_method
        self.deskew_processor = deskew_processor or DeskewProcessor()
        self.binarizer = BinarizationProcessor(method=binarization_method)
        self.noise_remover = NoiseRemovalProcessor()

    def process(self, image: np.ndarray) -> dict:
        """
        Process the image through the pipeline.

        Args:
            image (np.ndarray): Input image.

        Returns:
            dict: Dictionary containing the original, binarized, deskewed, and denoised images.
        """
        results = {
            'original_image': image,
            'binarized_image': None,
            'deskewed_image': None,
            'denoised_image': None,
        }
        results['binarized_image'] = self.binarizer.process(image)
        results['deskewed_image'] = self.deskew_processor.process(results['binarized_image'])
        results['denoised_image'] = self.noise_remover.process(results['deskewed_image'])
        return results


def GUI(image_path):
    img = io.imread(image_path)[...,:3]

    lineremoved = detect_vertical_line_and_crop(img)
    lineremoved = rgb2gray(lineremoved)

    binary =otsu(lineremoved)

    staffHeight1, spaceHeight1 =getRefLengths(binary)
    staffHeight1=2; spaceHeight1=16

    filteredImg1, candidates1 = getCandidateStaffs(binary, staffHeight1)
    filteredImg1, candidates1, eliminated1 = removeLonelyStaffs(candidates1, binary, staffHeight1, spaceHeight1, eliminated=[])

    without_staff = (binary-filteredImg1).astype(np.uint8)

    lines1 = getLines(1-filteredImg1, staffHeight1, spaceHeight1)
    staff1 = cv2.cvtColor(without_staff, cv2.COLOR_GRAY2BGR)
    for line in lines1:
        cv2.line(staff1,(0,line), (staff1.shape[1]-1, line), (0,0,255), staffHeight1)

    objects1 = segmentImage(without_staff, lines1, staffHeight1, spaceHeight1)
    cropped_objects = []

    for o in objects1:
        cropped = without_staff[o[1]:o[3], o[0]:o[2]]
        cropped_objects.append(cropped)

    cp1 = cv2.cvtColor(without_staff, cv2.COLOR_GRAY2BGR)
    for o in objects1:
        cv2.rectangle(cp1, (o[0],o[1]), (o[2],o[3]), (0, 255, 0), 3)

    file = open("/Users/malekelsaka/Desktop/Fall 2024/Projects/Image-Project/src/MLP.pickle",'rb')
    MLP = pickle.load(file)

    random_seed = 42
    random.seed(random_seed)
    np.random.seed(random_seed)

    classes = ['a_1', 'a_16', 'a_2', 'a_32', 'a_4', 'a_8',
            'barline ', 'chord', 'clef', '.', '&&', '##', '&', '', '#', r'\meter<"4/2"> ', r'\meter<"4/4"> ']


    firstTime = True
    output = ""
    perviousAccedental = ""
    idx = 0
    for o in objects1:
        features = extract_hog_features(without_staff[o[1]:o[3], o[0]:o[2]],(32,32))
        symbol_name = classes[np.argmax( MLP.predict_proba([features]))]

        if symbol_name == "a_2" or symbol_name == "a_4":
            if isHalf(without_staff[o[1]:o[3], o[0]:o[2]],spaceHeight1) :
                symbol_name = "a_2"
            else:
                symbol_name = "a_4"
        if symbol_name =="clef":
            if firstTime:
                firstTime = False
                output+= '['
            else:
                output+= '],\n['
        elif firstTime:
        # print('contin')
            continue
        # beam
        elif (o[2]-o[0]) > 4*spaceHeight1:
            try:
                output += getNoteCharacter(without_staff, o, "beam", lines1, staffHeight1, spaceHeight1)+" "
            except Exception as e:
                    print(f"Error occurred: {e}")
            continue
        # dot and barline
        elif symbol_name == "." or symbol_name == "barline ":
                if isDot(without_staff[o[1]:o[3], o[0]:o[2]],spaceHeight1):
                    output += "."

        # chord
        elif symbol_name == "chord":
        # print("chord")
            try:
                notes = getchordText([o[1],o[3]],without_staff[o[1]:o[3], o[0]:o[2]],staffHeight1,spaceHeight1,lines1)
            except:
                noteSymbol = getNoteCharacter(without_staff, o, "a_4", lines1, staffHeight1, spaceHeight1)

                output += noteSymbol[0]+perviousAccedental+noteSymbol[1:]+" "
                perviousAccedental = ""

                continue
            output +="{"
            for k in range(0,len(notes)-2,2):
                output += notes[k:k+2]+"/4,"
            output += notes[-2:]+"/4"
            output+= "} "

        # note
        elif symbol_name!="" and  symbol_name[0]=="a" :
            try:
                noteSymbol = getNoteCharacter(without_staff, o, symbol_name, lines1, staffHeight1, spaceHeight1)
                output += noteSymbol[0]+perviousAccedental+noteSymbol[1:]+" "
                perviousAccedental = ""
            except:
    #
                continue
        # accedentals
        elif symbol_name == r'\meter<"4/2"> ' or symbol_name == r'\meter<"4/4"> ':
            output += symbol_name
        else:

            perviousAccedental= symbol_name
    output+="]"
    if len(output.split("\n"))>1:
        output ="{\n"+output+"\n}"
    
    print('output',output)
    return cp1, output


def main():
    """
    Main function to demonstrate the image processing pipeline.
    """
    # _path = os.path.join(os.getcwd(), 'data', 'input')
    # for file in os.listdir(_path):
    #     if file.endswith(".jpg") or file.endswith(".PNG"):
    #         image_path = os.path.join(_path, file)
    #         image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    #         if image is None:
    #             raise FileNotFoundError(f"Image file not found at {image_path}")
    #         pipeline = ImageProcessingPipeline()
    #         results = pipeline.process(image)
    #         img_path = image_path.replace("input", "output")
    #         img_path = img_path.replace(img_path.split("/")[-1], image_path.split("/")[-1].replace(".jpg", "_clean.jpg"))
    #         cv2.imwrite(img_path, (results['denoised_image']))


if __name__ == "__main__":
    main()

