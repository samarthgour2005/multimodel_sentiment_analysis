# multimodal_sentiment_analysis

**What It Does**
- **Purpose:** Implements and compares text-only, image-only, and multimodal (text+image) sentiment classifiers on the MVSA (MVSA_Single) dataset.
- **Outputs:** Trained models, accuracy and loss plots, classification reports, confusion matrices, and a comparison chart between baselines and the fusion model.

**How It Works**
- **Data loading:** Loads text files and matching images from the MVSA_Single folder and reads labels from `labelResultAll.txt`. Conflicting text/image labels are resolved with a simple rule that favors non-neutral labels when disagreement occurs.
- **Preprocessing:** Uses a BERT tokenizer for text (max length 64) and torchvision transforms for images (resize/crop, normalization, augmentation for training).
- **Datasets & Loaders:** `MVSADataset` wraps text, images and labels; `DataLoader` provides batches for training and validation.
- **Models:**
	- **TextModel:** `bert-base-uncased` with a linear classifier on pooled output.
	- **ImageModel:** Pretrained `resnet18` fine-tuned with a 3-way output.
	- **FusionModel:** Uses BERT and ResNet18 as feature extractors, projects features to common dimension, concatenates them and applies a feed-forward head for 3-class sentiment prediction.
- **Training & Evaluation:** Training loops (`train_model`, `train_and_collect`) and `evaluate` functions compute losses, accuracies, classification reports, and confusion matrices; plotting utilities visualize metrics.
- **Baselines:** TF-IDF vectorizer with Logistic Regression and an XGBoost classifier are trained on text-only features as classical baselines.

**Used Technologies & Libraries**
- **Core:** Python, NumPy, pandas
- **Deep learning:** PyTorch, torchvision
- **Transformers:** Hugging Face `transformers` (BERT tokenizer & model)
- **Classical ML:** scikit-learn (TF-IDF, metrics, train/test split), XGBoost
- **Utilities & viz:** PIL (Pillow), matplotlib, seaborn, tqdm

**Dataset**
- **Name:** MVSA_Single (multimodal sentiment analysis dataset)
- **Location (in notebook):** `/kaggle/input/datasets/vincemarcs/mvsasingle/MVSA_Single`
- **Files used:** image files (`.jpg`), text files (`.txt`) in `data`, and `labelResultAll.txt` for labels.
- **Labels:** negative, neutral, positive (mapped to integers 0,1,2).

**How to Run**
- Install dependencies (example):

```bash
python -m pip install -r requirements.txt
```

- Open and run the notebook: [multimodal-sentiment-analysis.ipynb](multimodal-sentiment-analysis.ipynb#L1)
- Key cells to run in order: data loading, preprocessing, dataset creation, model definitions, training loops, evaluation and plotting.

**Notes & Tips**
- Training BERT+ResNet fusion requires a GPU for reasonable training speed.
- Adjust batch sizes and learning rates to fit your hardware.
- If using a different dataset path, update the `text_dir` and `label_file` variables in the notebook.

**Model Comparison Image**

  <img src="assets\model_comparsion.png" width="500">