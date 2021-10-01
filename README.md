# An Automated Cell Line Authentication Method for AstraZeneca Global Cell Bank

This is the official public Pytorch implementation for our paper (paper link), which was submitted to Nature Laboratory Investigation.


For any issue and question, please email [lt228@leicester.ac.uk]


## Dependencies

- Python (>=3.6)
- Pytorch (>=1.9.0)
- opencv-python
- matplotlib
- scikit-learn (>=0.24.2)
- numpy
- scikit-image
- torchvision (>=0.10.0)

## Dataset

Part of example images are put in the './data'. The whole dataset will be published after getting license.

<img src="./figures/Fig. 4.jpeg" alt="centered image" width="700" height="700">

## Training CLCNet

```bash
cd ./networks/classification
python cell_classification.py --bs 20 --arch Xception 
```

## Testing CLCNet
```bash
cd ./networks/classification
python cell_classification.py --bs 20 --arch Xception --resume ./checkpoint/Xception/best.pth --evaluate
```
## Training/Testing CLRNet or with tranfer learning 
Use JupyterLab to open 'model_evaluation.ipynb' and run all code blocks.

## License
This code is made available under the GPLv3 License and is available for non-commercial academic purposes.

## Acknowledgement
The authors gratefully acknowledge financial support from University of Leicester, AstraZeneca UK, China Scholarship Council.

------
If you find that is useful in your research, please consider citing:
```

```
