# Decomposed Attention Frequency debiased transformer model (DAF) 
This is an official implementation of a KDD 2026 paper, "Decopmosed Attention Frequency Debiased Transformer Model: Large Time-series Prediction Model For Satellite Orbit Prediction"

# Contribution 
DAF focuses on addressing the issue of limited generalization capability in existing time-series models for satellite orbit prediction across diverse orbital regimes and satellite types. This limitation can cause models to fail when applied to satellites with different characteristics or orbital parameters from their training data. We have conducted comprehensive zero-shot evaluations on this challenge and proposed a novel solution through architectural modifications including RFFT transformation, positional embedding adoption, and Tensor Train Decomposition integration. Extensive experimental results on seven constellation datasets and three real-world satellite datasets demonstrate the effectiveness and deployment potential of DAF.

# Architecture
<div align="center">
<img src="assets/DAF Architecture.PNG" width="1400">
</div>

# Dependencies
First, please make sure you have installed Python and pip. Then, the required dependencies can be installed by:
```bash
pip install -r requirements.txt
```

# Data Preparation
Next, collect the Starlink TLE dataset from Space-Track (link: https://www.space-track.org/). Subsequently, install the Orekit library (link: https://www.orekit.org/). The collected TLE data is then processed using Orekit to perform interpolation at 1-minute intervals, obtaining the final dataset. Please convert the 1-minute interpolated data to CSV format and then convert it to .pt format.


# Training
The data is prepared, you can train the model using the following command:
```bash
python main.py
```

# Inference
After training is completed, the model is saved in the current directory in the format best_{model}({seed}).pt. Subsequently, collect TLE data for ASTROCAST, CAPELLA, KINEIS, KUIPER, ICEYE, LEMUR, and SKYSAT from the Space-Track website and interpolate them using the Orekit library. Using this data, execute the following command:
```bash
python inference.py
```

# Experiment Results
Here are the main results of our experiment:

<img src="assets/Main Results.png" width="1000">

Also, zero-shot evaluation of 7 constellation datasets and 3 real-world satellite datasets.

<img src="assets/Zero-shot Evaluation.PNG" width="1400">


