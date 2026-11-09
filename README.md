# GI Endoscopy Image Classification

A deep learning project that looks at a gastrointestinal endoscopy image and decides which of 8 categories it belongs to. It also reports how confident it is, so that doubtful predictions can be sent to a person instead of being trusted blindly.

The project trains two models (ResNet101V2 and EfficientNetB3) on the public Kvasir dataset, tests them with a leakage-aware split, and includes a Streamlit app where you can upload an image and see the prediction, the confidence and a set of charts.

This is a research prototype. It is not a medical device and must not be used for clinical decisions.

## Why this exists

Endoscopy produces thousands of images, and reading them is slow and tiring. A model that can sort images and flag the uncertain ones could help a specialist spend their time where it matters.

Plenty of papers report accuracy on this dataset, but many of them split the data randomly. Endoscopy frames taken moments apart look almost identical, so a random split can put a near copy of a test image into the training set. The model then looks better than it really is. The main point of this project is a more careful evaluation:

- near-duplicate images are found with perceptual hashing and kept together on one side of the split
- the test set is used once, at the very end
- results come with confidence intervals, macro-F1 and a calibration measure (ECE), not just accuracy

## The 8 classes

dyed-lifted-polyps, dyed-resection-margins, esophagitis, normal-cecum, normal-pylorus, normal-z-line, polyps, ulcerative-colitis

The dataset has 4,000 images, 500 per class, split 70 / 15 / 15 into training, validation and testing (603 test images).

## What is in this repository

| File | What it does |
| --- | --- |
| `gi_pipeline.ipynb` | The full training pipeline as a notebook (one code cell). Built for Kaggle. |
| `gi_pipeline_cell.py` | The same pipeline as a plain Python file. |
| `app.py` | Streamlit dashboard: upload an image, get the prediction, confidence and charts. |
| `test_dashboard.py` | A smaller Streamlit page that only shows the prediction and confidence for one image. |
| `test.py` | A command line test. Opens a file picker, then prints the predicted class and confidence. |
| `requirements.txt` | Python packages needed to run the apps. |

The trained models (`*.keras`) are not stored in the repository because they are large and GitHub rejects files over 100 MB. See the section on models below.

## How the training works

1. The notebook downloads the dataset from Kaggle automatically.
2. Every image gets a 64-bit perceptual hash. Images whose hashes differ by 4 bits or fewer are linked into groups.
3. A stratified, group-aware split (20 folds, seed 42) puts each group entirely into training, validation or test. The code checks that no group appears in two of them.
4. Each model starts from ImageNet weights and gets a new head: pooling, batch norm, dropout, a 256 unit swish layer, dropout, then an 8-way softmax.
5. Stage 1 trains only the head with the backbone frozen (learning rate 1e-3, up to 10 epochs).
6. Stage 2 unfreezes the backbone except the batch norm layers and trains at a lower rate (1e-4, up to 40 epochs) with early stopping and learning rate reduction.
7. Other settings: AdamW with weight decay 1e-4, label smoothing 0.05, data augmentation on training images only, mixed precision, batch size 32 per GPU.
8. The test set is evaluated once, plainly and with flip test-time augmentation. The notebook saves the models, training curves, confusion matrices, the split file and `results.json`, then zips everything into `gi_results.zip`.

## Results

One run, one split, tested on 603 unseen images.

| | ResNet101V2 | EfficientNetB3 |
| --- | --- | --- |
| Accuracy (plain) | 93.70% | 93.53% |
| Accuracy (flip TTA) | 94.03% | 93.86% |
| 95% interval (flip TTA) | 91.85 to 95.66 | 91.66 to 95.52 |
| Macro-F1 (flip TTA) | 0.9398 | 0.9384 |
| ECE (plain) | 0.0308 | 0.0329 |

The two models are effectively tied. They differ by one image out of 603 and the intervals overlap almost completely. EfficientNetB3 has roughly four times fewer parameters (about 12 million against about 44 million).

Flip averaging raised accuracy slightly but made the confidence values a little less well calibrated.

## Running it

You need Python 3.10 or newer.

```
pip install -r requirements.txt
```

If loading a model fails with an error about `quantization_config`, upgrade Keras and TensorFlow. The models were trained on a newer Keras than some local installs have.

```
pip install --upgrade keras tensorflow
```

### Training

Open `gi_pipeline.ipynb` on Kaggle, turn on a GPU, and run the cell. It downloads the data and trains both models. Outputs are written to `/kaggle/working/gi_results` and zipped for download.

### Models

Download `gi_results.zip` from your Kaggle run, extract it, and place the `*_final.keras` files (or the whole `gi_results` folder) next to `app.py`.

### The dashboard

```
streamlit run app.py
```

Pick the model or models in the sidebar, upload an image, and open the tabs. If you select both models, their probabilities are averaged. You can also set a review threshold: predictions with confidence below it are flagged for human review.

### The simple tests

```
streamlit run test_dashboard.py
python test.py
```

`test.py` opens a file dialog, so it needs a desktop environment.

## Limitations

- One dataset, one split and one training run per model, so run to run variation was not measured.
- No testing on images from other hospitals, cameras or patients.
- The model only knows 8 classes. An unrelated image will still get one of them, usually with lower confidence.
- Near-duplicates are grouped, not removed, so look-alike frames still exist inside the test set. The true uncertainty may be a bit wider than the reported intervals.
- No clinical validation.

## Dataset and credits

Kvasir: Pogorelov, K., et al. (2017). KVASIR: A multi-class image dataset for computer aided gastrointestinal disease detection. Proceedings of ACM MMSys. Please check the dataset's own license and terms before reusing it.

Models are the Keras implementations of ResNet101V2 and EfficientNetB3 with ImageNet weights.

## Author

[Your name], [your organization], [your email]

## License

Add a license of your choice before publishing, for example MIT for the code. This does not cover the dataset.
