import random
import copy
import numpy as np
import torchvision.transforms.functional as TF

from torch.utils import data as data
from torchvision.transforms.functional import normalize

from basicsr.data.data_util import paired_paths_from_folder, paired_paths_from_meta_info_file
from basicsr.data.transforms import augment, paired_random_crop
from basicsr.utils import FileClient, imfrombytes, img2tensor, padding

try:
    from basicsr.utils.registry import DATASET_REGISTRY
except Exception:
    DATASET_REGISTRY = None


def _register_dataset(cls):
    if DATASET_REGISTRY is not None:
        return DATASET_REGISTRY.register()(cls)
    return cls


@_register_dataset
class MultiPairedImageDataset(data.Dataset):
    def __init__(self, opt):
        super(MultiPairedImageDataset, self).__init__()
        self.opt = opt
        self.file_client = None
        self.io_backend_opt = copy.deepcopy(opt['io_backend'])
        self.mean = opt['mean'] if 'mean' in opt else None
        self.std = opt['std'] if 'std' in opt else None
        self.filename_tmpl = opt.get('filename_tmpl', '{}')

        self.task_names = opt['task_names']
        assert isinstance(self.task_names, (list, tuple)), \
            'task_names should be a list, e.g., [rain, haze, snow].'

        self.gt_roots = {}
        self.lq_roots = {}

        for task_name in self.task_names:
            gt_key = f'dataroot_gt_{task_name}'
            lq_key = f'dataroot_lq_{task_name}'

            assert gt_key in opt, f'{gt_key} is not found in dataset options.'
            assert lq_key in opt, f'{lq_key} is not found in dataset options.'

            self.gt_roots[task_name] = opt[gt_key]
            self.lq_roots[task_name] = opt[lq_key]

        self.paths = []
        self.task_path_dict = {}

        if self.io_backend_opt['type'] == 'lmdb':
            raise NotImplementedError(
                'MultiPairedImageDataset currently supports folder/meta_info_file mode. '
                'LMDB mode is not implemented for multi-task training.'
            )

        for task_name in self.task_names:
            assert task_name in self.gt_roots, f'{task_name} not found in dataroot_gt.'
            assert task_name in self.lq_roots, f'{task_name} not found in dataroot_lq.'

            gt_folder = self.gt_roots[task_name]
            lq_folder = self.lq_roots[task_name]

            task_filename_tmpl = self._get_task_value(self.filename_tmpl, task_name, default='{}')

            meta_info_file = None
            if 'meta_info_file' in self.opt and self.opt['meta_info_file'] is not None:
                meta_info_file = self._get_task_value(self.opt['meta_info_file'], task_name, default=None)

            if meta_info_file is not None:
                task_paths = paired_paths_from_meta_info_file(
                    [lq_folder, gt_folder], ['lq', 'gt'], meta_info_file, task_filename_tmpl)
            else:
                task_paths = paired_paths_from_folder(
                    [lq_folder, gt_folder], ['lq', 'gt'], task_filename_tmpl)

            for item in task_paths:
                item['task'] = task_name

            self.task_path_dict[task_name] = task_paths
            self.paths.extend(task_paths)

        if len(self.paths) == 0:
            raise RuntimeError('No paired images found for MultiPairedImageDataset.')

        random.shuffle(self.paths)

    def _get_task_value(self, value, task_name, default=None):
        if isinstance(value, dict):
            return value.get(task_name, default)
        return value

    def __getitem__(self, index):
        if self.file_client is None:
            io_backend_opt = copy.deepcopy(self.io_backend_opt)
            self.file_client = FileClient(io_backend_opt.pop('type'), **io_backend_opt)

        scale = self.opt['scale']
        index = index % len(self.paths)

        gt_path = self.paths[index]['gt_path']
        lq_path = self.paths[index]['lq_path']
        task_name = self.paths[index].get('task', 'unknown')

        img_bytes = self.file_client.get(gt_path, 'gt')
        try:
            img_gt = imfrombytes(img_bytes, float32=True)
        except Exception:
            raise Exception(f'gt path {gt_path} not working.')

        img_bytes = self.file_client.get(lq_path, 'lq')
        try:
            img_lq = imfrombytes(img_bytes, float32=True)
        except Exception:
            raise Exception(f'lq path {lq_path} not working.')

        if self.opt['phase'] == 'train':
            gt_size = self.opt['gt_size']

            img_gt, img_lq = padding(img_gt, img_lq, gt_size)

            img_gt, img_lq = paired_random_crop(
                img_gt, img_lq, gt_size, scale, gt_path)

            img_gt, img_lq = augment(
                [img_gt, img_lq],
                self.opt.get('use_flip', False),
                self.opt.get('use_rot', False)
            )

        img_gt, img_lq = img2tensor(
            [img_gt, img_lq],
            bgr2rgb=True,
            float32=True
        )

        if self.opt['phase'] == 'train':
            aug = random.randint(0, 2)
            if aug == 1:
                img_lq = TF.adjust_gamma(img_lq, 1)
                img_gt = TF.adjust_gamma(img_gt, 1)

            aug = random.randint(0, 2)
            if aug == 1:
                sat_factor = 1 + (0.2 - 0.4 * np.random.rand())
                img_lq = TF.adjust_saturation(img_lq, sat_factor)
                img_gt = TF.adjust_saturation(img_gt, sat_factor)

        if self.mean is not None or self.std is not None:
            normalize(img_lq, self.mean, self.std, inplace=True)
            normalize(img_gt, self.mean, self.std, inplace=True)

        return {
            'lq': img_lq,
            'gt': img_gt,
            'lq_path': lq_path,
            'gt_path': gt_path,
            'task': task_name
        }

    def __len__(self):
        return len(self.paths)