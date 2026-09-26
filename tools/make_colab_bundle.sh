#!/usr/bin/env bash
# pack code + features for Colab/Kaggle (upload the zip to your Drive)
set -e
cd "$(dirname "$0")/.."
rm -f colab_bundle.zip
zip -qr colab_bundle.zip webswipe -x '*__pycache__*'
zip -qj colab_bundle.zip data/*_features.npz
ls -lh colab_bundle.zip
