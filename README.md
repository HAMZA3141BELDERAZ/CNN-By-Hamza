# Digit Lab: MNIST CNN + Streamlit

A handwritten-digit classifier based on the TensorFlow CNN in `CNN.ipynb`. The app trains the model on MNIST, accepts a PNG or JPEG, and displays the predicted digit with confidence scores for all ten classes.

## Run locally

Python 3.10 or newer is recommended. In PowerShell, from this project folder:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
streamlit run app.py
```

Select **Train model** in the sidebar on first launch. Training downloads MNIST, runs five epochs, evaluates on the test split, and saves the model under `models/`. The model artifact is intentionally ignored by Git; it can be regenerated from the app.

For useful predictions, upload one clear digit centered on a simple background. The preprocessing converts the image to grayscale, normalizes its polarity, crops the foreground, and centers it in a 28x28 input.

## GitHub

Install Git and authenticate with GitHub CLI (`gh auth login`), then run from this folder:

```powershell
git init
git add .
git commit -m "Build MNIST CNN Streamlit app"
gh repo create mnist-cnn-streamlit --public --source=. --remote=origin --push
```

To deploy, push the repository to GitHub and create a Streamlit Community Cloud app pointing to `app.py`. The first user must click **Train model**; the trained weights are saved locally at runtime and are not committed.

## Project files

- `app.py`: Streamlit interface, training flow, image preprocessing, and prediction.
- `cnn_model.py`: CNN architecture and compile settings.
- `requirements.txt`: Python dependencies.
