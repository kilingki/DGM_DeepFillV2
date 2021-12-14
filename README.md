## DeepFillv2_Pytorch
This is a Pytorch re-implementation for the paper [Free-Form Image Inpainting with Gated Convolution](https://arxiv.org/abs/1806.03589).  
This is a Team Project Codes for DGM/ACV Class.

This repository contains "Gated Convolution", "Contextual Attention" and "Spectral Normalization".
## Requirement
- Python 3
- OpenCV-Python
- Numpy
- Pytorch 1.0+

## Dataset
### Training Dataset
The original training dataset is a collection of images from [Places365-Standard](http://places2.csail.mit.edu/download.html) which spatial sizes are larger than 512 * 512. (It will be more free to crop image with larger resolution during training)

For Class, make 'dgm_dataset' folder and move training images in this folder.

### Testing Dataset
Create the folders `dgm_images` and `dgm_masks`. 
Note that 2 folders contain the image and its corresponding mask respectively.
Testing dataset is downloaded from DGM Class.

## Training
* To train a model:
``` bash
$ bash ./run_train.sh
``` 
All training models and sample images will be saved in `./models/` and `./samples/` respectively.
## Testing
Get trained generator pth file and put it in `./pretrained_model/`.
* To test a model:
``` bash
$ bash ./run_test.sh
``` 
## Acknowledgments
The main code is based upon [deepfillv2](https://github.com/zhaoyuzhi/deepfillv2) and [deepfillv2](https://github.com/csqiangwen/DeepFillv2_Pytorch).
The code of "Contextual Attention" is based upon [generative-inpainting-pytorch](https://github.com/daa233/generative-inpainting-pytorch).  
Thanks for their excellent works!  
And Thanks for [Kuaishou Technology Co., Ltd](https://www.kuaishou.com/en) providing the hardware support to me.
## Citation
```
@article{yu2018generative,
  title={Generative Image Inpainting with Contextual Attention},
  author={Yu, Jiahui and Lin, Zhe and Yang, Jimei and Shen, Xiaohui and Lu, Xin and Huang, Thomas S},
  journal={arXiv preprint arXiv:1801.07892},
  year={2018}
}

@article{yu2018free,
  title={Free-Form Image Inpainting with Gated Convolution},
  author={Yu, Jiahui and Lin, Zhe and Yang, Jimei and Shen, Xiaohui and Lu, Xin and Huang, Thomas S},
  journal={arXiv preprint arXiv:1806.03589},
  year={2018}
}
```
