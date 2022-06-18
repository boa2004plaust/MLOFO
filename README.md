# MLOFO
It is for "Efficient multi-granularity network for fine-grained image classification".

Author: Jiabao Wang, Yang Li, Hang Li, Xun Zhao, Rui Zhang, Zhuang Miao.

Last Update: 18/06/2022

CITATION:

If you use this code in your research, please cite:

	@ARTICLE {MLOFO,
	author    = "Jiabao Wang, Yang Li, Hang Li, Xun Zhao, Rui Zhang, Zhuang Miao",
	title     = "Efficient multi-granularity network for fine-grained image classification",
	journal   = {Journal of Real-Time Image Processing},
	year      = {2022},
	}
  
## 1. Code Usage

***Notes: All of the codes we support work well dependent on Python3.7 & Pytorch 1.7.1 with four GTX 1080Ti GPUs***

  ### Training
   ```Shell
   ./train_MLOFO.sh
   ```
  ### Testing
   ```Shell
   ./test_MLOFO.sh
   ```

  Before running the code, you should modify the dataset path. The pretrained model for CUB can be download from [https://pan.baidu.com/s/1oTgI2QiH8j5VdCRasg07Wg](code：2022).
