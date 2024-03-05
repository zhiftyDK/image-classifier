# **Image Classifier**

### **Install**
```bash
npm install @zhiftydk/image-classifier
```

### **About**
This nodejs package is the easiest image classifier to use. The package is based on tensorflows knn-classifier and mobilenet model.

### **How to use**
**Dateset/image directory configuration:**
```
Animals/
... cats/
    ... cat1.jpg
    ... cat2.jpg
    ... cat3.jpg
... dogs/
    ... dog1.jpg
    ... dog2.jpg
    ... dog3.jpg
```

In this example `animals/` is the parent folder, and all the subdirectories like e.g. `dogs/` & `cats/` will become the different classes used in the classification.

**Import:**
```js
// CommonJS
const ImageClassifier = require("@zhiftydk/image-classifier");

// Or ES Modules
import ImageClassifier from "@zhiftydk/image-classifier";
```

**Example:**
```js
// OPTIONAL: Set image size. Default = 64
ImageClassifier.imageSize = 64;

// Process images (Generates a model.json in the root directory)
const datasetPath = "./animals/";
ImageClassifier.process(datasetPath); // Returns nothing

// Compare a new image with the model and classify it
const newImagePath = "./testimage.jpg"
ImageClassifier.compare(newImagePath).then(result => {
    console.log(result);
});
```