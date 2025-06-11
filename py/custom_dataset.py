import cv2
import numpy as np
from PIL import Image

from ultralytics.data.dataset import ClassificationDataset
from ultralytics.utils import LOGGER


class CustomClassificationDataset(ClassificationDataset):
    """
    这是一个自定义的数据集类，它继承自 Ultralytics 的 ClassificationDataset，
    但重写了 __getitem__ 方法以使用 Pillow (`Image.open`) 而不是 OpenCV (`cv2.imread`)
    来加载图片。这旨在解决 `cv2.imread` 在处理某些特定图片时可能发生的挂起问题。
    """

    def __getitem__(self, i: int) -> dict:
        """
        通过使用 Pillow 加载图片来返回样本。

        Args:
            i (int): 样本索引。

        Returns:
            (dict): 包含图像张量和类别索引的字典。
        """
        f, j, fn, im = self.samples[i]  # filename, index, filename.with_suffix('.npy'), image

        # 核心修改：使用 Pillow (Image.open) 替代 OpenCV (cv2.imread)
        try:
            # 直接使用 Pillow 加载图片并转换为 RGB
            # 这可以避免 cv2.imread 挂起和后续的 BGR->RGB 转换
            im = Image.open(f).convert("RGB")
        except Exception as e:
            LOGGER.warning(f"WARNING ⚠️ Error loading image {f} with Pillow: {e}")
            # 如果一张图片损坏，返回数据集中另一张随机图片的副本
            return self.__getitem__(np.random.randint(len(self)))

        # 缓存逻辑保持不变，但现在缓存的是 Pillow 加载的、已经是 NumPy 数组的图像
        # 注意：原始代码的缓存逻辑可能仍然依赖于 cv2，这里我们简化处理，
        # 总是从磁盘加载。如果需要缓存，需要更复杂的逻辑来处理 Pillow 对象。
        # 为了调试，我们暂时忽略缓存，以确保核心加载逻辑正确。

        # if self.cache_ram and im is None:
        #     # 对于 RAM 缓存，我们已经加载了 im
        #     self.samples[i][3] = np.array(im)
        # elif self.cache_disk:
        #     if not fn.exists():
        #         np.save(fn.as_posix(), np.array(im), allow_pickle=False)
        #     im = np.load(fn)

        # 应用变换
        sample = self.torch_transforms(im)
        return {"img": sample, "cls": j} 
