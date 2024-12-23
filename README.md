# Image Processing and Classification Project
This project focuses on the development of an advanced image processing pipeline for musical symbol recognition. The goal is to automate the extraction and classification of musical symbols
from images of sheet music, enabling their digital representation for further processing or analysis.Sheet music images typically contain various symbols, staff lines, and other elements that need 
to be accurately segmented and classified to interpret the underlying musical content.

You can simply run this project by runing the gui.py file.

Below is a list of the main libraries used in this project along with the specific functions and methods utilized.

## Libraries Used

### Standard Libraries
- **os**
  - `os.listdir()`
  - `os.path.join()`
  - `os.path.isdir()`
- **random**
  - `random.seed()`
- **pickle**
  - `pickle.load()`
  - `pickle.dump()`
- **joblib**
  - `joblib.load()`
  - `joblib.dump()`
- **traceback**
  - `traceback.print_exc()`

### Numerical and Scientific Libraries
- **numpy**
  - `np.ndarray`
  - `np.random.seed()`
  - `np.argmax()`
  - `np.float32`
  - `np.uint8`
- **scipy**
  - `scipy.signal.convolve2d()`
  - `scipy.signal.find_peaks()`
  - `scipy.signal.peak_widths()`
  - `scipy.fftpack.fft2()`
  - `scipy.fftpack.ifft2()`

### Image Processing Libraries
- **cv2** (OpenCV)
  - `cv2.imread()`
  - `cv2.cvtColor()`
  - `cv2.threshold()`
  - `cv2.HOGDescriptor()`
  - `cv2.resize()`
  - `cv2.rectangle()`
  - `cv2.IMREAD_GRAYSCALE`
  - `cv2.COLOR_GRAY2BGR`
  - `cv2.THRESH_BINARY`
  - `cv2.THRESH_OTSU`
  - `cv2.imwrite()`
- **skimage** (scikit-image)
  - `skimage.io.imread()`
  - `skimage.color.rgb2gray()`
  - `skimage.filters.threshold_otsu()`
  - `skimage.filters.gaussian()`
  - `skimage.filters.median()`
  - `skimage.morphology.binary_opening()`
  - `skimage.morphology.binary_closing()`
  - `skimage.morphology.binary_dilation()`
  - `skimage.morphology.binary_erosion()`
  - `skimage.morphology.closing()`
  - `skimage.morphology.opening()`
  - `skimage.morphology.square()`
  - `skimage.morphology.skeletonize()`
  - `skimage.morphology.disk()`
  - `skimage.morphology.thin()`
  - `skimage.transform.resize()`
  - `skimage.transform.probabilistic_hough_line()`
  - `skimage.transform.hough_line()`
  - `skimage.transform.hough_line_peaks()`
  - `skimage.util.random_noise()`
  - `skimage.exposure.histogram()`
  - `skimage.feature.canny()`

### Machine Learning Libraries
- **torch** (PyTorch)
  - `torch.tensor()`
  - `torch.device()`
  - `torch.cuda.is_available()`
  - `torch.cdist()`
  - `torch.topk()`
  - `torch.mode()`
  - `torch.nn.Linear()`
  - `torch.optim.SGD()`
  - `torch.nn.CrossEntropyLoss()`
  - `torch.utils.data.DataLoader()`
  - `torch.utils.data.TensorDataset()`
- **sklearn** (scikit-learn)
  - `sklearn.model_selection.train_test_split()`
  - `sklearn.metrics.f1_score()`
  - `sklearn.metrics.recall_score()`
  - `sklearn.metrics.classification_report()`
  - `sklearn.metrics.accuracy_score()`
  - `sklearn.cluster.KMeans()`
  - `sklearn.svm.SVC()`
  - `sklearn.model_selection.GridSearchCV()`

### Visualization Libraries
- **matplotlib**
  - `matplotlib.pyplot.imshow()`
  - `matplotlib.pyplot.show()`
  - `matplotlib.pyplot.figure()`
  - `matplotlib.pyplot.subplot()`
  - `matplotlib.pyplot.title()`
  - `matplotlib.pyplot.bar()`
  - `matplotlib.pyplot.cm()`
  - `matplotlib.pyplot.Axes3D()`
  - `matplotlib.pyplot.LinearLocator()`
  - `matplotlib.pyplot.FormatStrFormatter()`

### Typing and Abstract Base Classes
- **typing**
  - `typing.Union`
  - `typing.Tuple`
  - `typing.List`
  - `typing.Optional`
- **abc**
  - `abc.ABC`
  - `abc.abstractmethod`

## Installation

To install the required libraries, you can use the following commands:

```sh
pip install numpy scipy opencv-python scikit-image torch torchvision torchaudio scikit-learn matplotlib
