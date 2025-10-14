# BERT Fine-tuning Guide for Patent Classification

This guide explains how to fine-tune BERT models on the patent classification dataset using both local and Modal-based training.

## 🚀 Quick Start

### Option 1: Local Training (CPU/GPU)

```bash
# Basic training with default settings
python train.py

# Custom training with specific parameters
python train.py \
  --model_name bert-base-multilingual-uncased \
  --num_epochs 5 \
  --batch_size 32 \
  --learning_rate 3e-5 \
  --max_samples 1000

# Push trained model to HuggingFace Hub
python train.py \
  --push_to_hub \
  --hub_model_id your-username/patent-bert-classifier
```

### Option 2: Modal Training (GPU Accelerated)

```bash
# Deploy and run training on Modal
modal run train_modal.py

# Or use the Modal CLI with parameters
modal run train_modal.py --model_name bert-base-uncased
```

## 📋 Training Options

### Model Selection
- `bert-base-uncased`: General English BERT (recommended for English patents)
- `bert-base-multilingual-uncased`: Supports multiple languages
- `bert-large-uncased`: Larger model, better accuracy but slower
- `distilbert-base-uncased`: Faster, smaller model

### Training Parameters
- `--num_epochs`: Number of training epochs (default: 3)
- `--learning_rate`: Learning rate (default: 2e-5)
- `--batch_size`: Training batch size (default: 16)
- `--max_samples`: Limit samples for testing (default: None - use all)
- `--early_stopping_patience`: Early stopping patience (default: 3)

### Output Options
- `--output_dir`: Where to save the trained model
- `--push_to_hub`: Upload model to HuggingFace Hub
- `--hub_model_id`: Model ID on Hub (required if push_to_hub)

## 🎯 Expected Performance

Based on the patent classification dataset, you can expect:

| Model | Expected Accuracy | Training Time (3 epochs) |
|-------|------------------|---------------------------|
| BERT-base | 75-85% | 30-60 minutes (GPU) |
| BERT-large | 80-90% | 60-120 minutes (GPU) |
| DistilBERT | 70-80% | 15-30 minutes (GPU) |

## 📊 Usage Examples

### Example 1: Quick Test Training
```bash
# Train on a small subset for testing
python train.py --max_samples 1000 --num_epochs 1
```

### Example 2: Full Production Training
```bash
# Full training with multilingual BERT
python train.py \
  --model_name bert-base-multilingual-uncased \
  --num_epochs 5 \
  --batch_size 32 \
  --learning_rate 3e-5 \
  --output_dir ./models/patent-bert-multilingual \
  --push_to_hub \
  --hub_model_id your-username/patent-bert-multilingual
```

### Example 3: Modal GPU Training
```bash
# Fast GPU training on Modal
modal run train_modal.py
```

## 🔧 Using Your Trained Model

After training, use your model with the main classification pipeline:

```bash
# Use locally saved model
python main.py --mode classify \
  --model ./models/patent-bert_20241010_143000 \
  --model_type classification \
  --max_samples 100

# Use model from HuggingFace Hub
python main.py --mode classify \
  --model your-username/patent-bert-classifier \
  --model_type classification \
  --max_samples 100
```

## 📈 Comparing Models

After training multiple models, compare their performance:

```bash
# Compare costs and performance
python main.py --mode compare
```

## 🛠️ Troubleshooting

### Common Issues

1. **Out of Memory Error**
   - Reduce `--batch_size` to 8 or 4
   - Use a smaller model like DistilBERT
   - Use gradient checkpointing (automatically enabled in Modal)

2. **Slow Training**
   - Use Modal for GPU acceleration
   - Reduce `--max_samples` for testing
   - Use mixed precision training (automatically enabled in Modal)

3. **Low Accuracy**
   - Increase `--num_epochs` to 5-10
   - Try different learning rates (1e-5, 3e-5, 5e-5)
   - Use a larger model (bert-large)

### Hardware Requirements

**Local Training:**
- **CPU**: 8+ GB RAM, 4+ hours for full dataset
- **GPU**: 8+ GB VRAM (RTX 3080/4080, V100, A100)
- **Storage**: 5+ GB for model and dataset

**Modal Training:**
- Automatically uses A10G GPU (24GB VRAM)
- Fast training: ~30-60 minutes for full dataset
- No local hardware requirements

## 📊 Dataset Information

The `ccdv/patent-classification` dataset contains:
- **Training**: 25,000 patent abstracts
- **Validation**: 5,000 patent abstracts  
- **Test**: 5,000 patent abstracts
- **Classes**: 9 patent categories (0-8)
- **Language**: English
- **Average length**: ~100-200 tokens per abstract

## 🎉 Next Steps

1. **Train your first model**: Start with the basic command
2. **Experiment with parameters**: Try different learning rates and epochs
3. **Compare approaches**: Test BERT vs. Llama vs. sentence transformers
4. **Deploy your model**: Use it in the main classification pipeline
5. **Share your model**: Push to HuggingFace Hub for others to use

---

## 💡 Pro Tips

- **Start small**: Use `--max_samples 1000` for quick experiments
- **Use Modal**: Much faster than local training, no setup required
- **Monitor training**: Check the logs for overfitting signs
- **Save everything**: Models are automatically timestamped
- **Compare results**: Use the results comparison tool after training