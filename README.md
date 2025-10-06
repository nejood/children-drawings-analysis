# Children's Drawing Analysis Project

A comprehensive machine learning pipeline for analyzing children's drawings using computer vision and natural language processing techniques.

## 🎯 Project Overview

This project combines multiple AI models to analyze children's drawings:
- **Emotion Classification**: Using Vision Transformer (ViT) to classify emotions in drawings
- **Object Detection**: Using DETR (Detection Transformer) to identify objects and elements
- **Image Captioning**: Using BLIP-2 to generate descriptive captions
- **Report Generation**: Automated PDF report generation with analysis results

## 🏗️ Architecture

```
Raw Data → Enhancement → Normalization → Model Training → Inference → Reports
```

### Models Used:
- **Emotion Classification**: `google/vit-base-patch16-224-in21k` with LoRA fine-tuning
- **Object Detection**: `facebook/detr-resnet-50` 
- **Image Captioning**: `Salesforce/blip2-opt-2.7b`

## 📁 Project Structure

```
children_drawings_project/
├── data/
│   ├── raw/                    # Original datasets
│   └── processed/              # Cleaned and processed data
├── scripts/                    # Main processing scripts
│   ├── enhance_images.py       # Image enhancement
│   ├── combine_emotions.py     # Dataset combination
│   ├── clean_normalize.py      # Data cleaning
│   ├── train_emotions_vit_hf.py # Emotion model training
│   ├── train_detection_detr_hf.py # Detection model training
│   ├── caption_blip2_infer_hf.py # Caption generation
│   ├── infer_and_report.py     # Full inference pipeline
│   └── report_utils.py         # PDF report utilities
├── outputs/                    # Generated results
├── Commands to run.txt         # Pipeline execution commands
└── deps.txt                   # Dependencies list
```

## 🚀 Quick Start

### 1. Environment Setup

```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install timm transformers peft accelerate datasets pyyaml pillow reportlab tqdm scikit-learn pandas
```

### 2. Data Preparation

```bash
# Enhance images
python scripts/enhance_images.py --in data/raw/emotions --out data/raw/emotions_enh --autocontrast --median 3 --sharpen 0.8 --gamma 0.95 --resize 224

# Combine datasets
python scripts/combine_emotions.py "data/raw/emotions_enh" "data/raw/Emotion_Recognition_multiclass_enh"

# Clean and normalize
python scripts/clean_normalize.py --mode emotions --in data/processed/emotions_combined --out data/processed/emotions_clean --resize 224 --dedup
```

### 3. Model Training

```bash
# Train emotion classification model
python scripts/train_emotions_vit_hf.py \
  --data_root data/processed/emotions_clean \
  --model_id google/vit-base-patch16-224-in21k \
  --epochs 30 --batch_size 32 --lr 3e-5 --img_size 224 --lora

# Train object detection model
python scripts/train_detection_detr_hf.py \
  --coco_root data/processed/htp_dap_clean_coco \
  --model_id facebook/detr-resnet-50 \
  --epochs 50 --batch_size 4 --lr 2e-5
```

### 4. Generate Analysis

```bash
# Generate captions
python scripts/caption_blip2_infer_hf.py \
  --images_dir data/processed/htp_dap_clean/images \
  --out_csv data/processed/htp_dap_clean/captions.csv \
  --model_id Salesforce/blip2-opt-2.7b

# Run full inference pipeline
python scripts/infer_and_report.py \
  --images_dir data/processed/htp_dap_clean/images \
  --out_dir outputs \
  --emotions_model_dir data/processed/emotions_clean/models/<model_dir> \
  --detr_model_dir data/processed/htp_dap_clean_coco/models/<model_dir> \
  --blip2_model_id Salesforce/blip2-opt-2.7b
```

## 📊 Features

- **Multi-modal Analysis**: Combines vision and language understanding
- **Automated Pipeline**: End-to-end processing from raw images to reports
- **Enhanced Data Processing**: Image enhancement and normalization
- **Efficient Training**: Uses LoRA for parameter-efficient fine-tuning
- **Professional Reports**: Generates PDF reports with analysis results
- **Scalable Architecture**: Designed for batch processing of multiple images

## 🔧 Technical Details

### Data Processing
- Image enhancement (contrast, sharpening, gamma correction)
- Data deduplication and normalization
- YOLO to COCO format conversion for detection tasks

### Model Training
- LoRA (Low-Rank Adaptation) for efficient fine-tuning
- Mixed precision training support
- Automated model saving and versioning

### Inference
- Batch processing capabilities
- Confidence threshold filtering
- Multi-format output (JSON, PDF)

## 📈 Performance

The models achieve:
- **Emotion Classification**: Competitive accuracy on children's drawing emotion recognition
- **Object Detection**: Effective identification of drawing elements
- **Caption Generation**: Contextually relevant descriptions

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/new-feature`)
3. Commit your changes (`git commit -am 'Add new feature'`)
4. Push to the branch (`git push origin feature/new-feature`)
5. Create a Pull Request

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Hugging Face for pre-trained models and transformers library
- Facebook Research for DETR architecture
- Salesforce Research for BLIP-2 model
- Google Research for Vision Transformer

## 📧 Contact

For questions or collaboration opportunities, please open an issue on GitHub.
