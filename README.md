# Public-Opinion-Analysis

A Weibo (Chinese microblog) public-opinion analysis system built on **ALBERT**. It uses a Chinese ALBERT pretrained model to classify whether a Weibo post/comment is a valid opinion, and exposes an HTTP prediction endpoint.

## Features

- Loads the Chinese ALBERT pretrained model (`albert_small_zh_google`)
- Builds a binary classification model on top of `bert4keras` to tell whether a Weibo text is a valid opinion
- Provides a Flask HTTP endpoint `/isValidOpinion` for real-time prediction

## Project Structure

```
src/
├── config.py                        # Path configuration (data_dir / output_dir / model_dir)
├── weibo_opinions_monitoring.py     # Prediction service (Flask)
├── albert_small_zh_google/          # ALBERT Chinese pretrained model files
├── bert4keras/                      # bert4keras dependency library
├── log.log                          # Model training log
└── best_model.weights.debug         # Trained model weights
```

## Dependencies

- Python 3.7
- TensorFlow 1.x
- Keras / bert4keras
- Flask
- numpy

## Quick Start

```bash
# Start the prediction service (default port 8330)
python src/weibo_opinions_monitoring.py
```

### API

**GET /isValidOpinion?query=<text>** or **POST /isValidOpinion** (form field `query`)

Returns JSON:

```json
{
  "isValid": "0 or 1",
  "score": "predicted probability",
  "query": "original text"
}
```

Example:

```
GET http://127.0.0.1:8330/isValidOpinion?query=You stupid idiot, what the hell are you talking about
```

(This is an abusive comment, which the model classifies as **not** a valid opinion.)

## Model

- Pretrained model: `albert_small_zh_google` (Chinese ALBERT small model)
- Binary classification: whether a Weibo text is a valid opinion
- Training details in `src/log.log`: 10 epochs, final test accuracy ~0.73
