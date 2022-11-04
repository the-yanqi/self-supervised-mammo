import gin, logging, os, torch, itertools
from torch.utils.data import Dataset, Sampler
from . import loading_mammogram, duke
from bisect import bisect_left

from PIL import Image
import numpy as np
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)

def img_dl_pos_func(datapac):
    # A datum is considered positive if it has either matched pathology or annotations
    return len(datapac["lesions"]) > 0

def load_mammogram_img(meta_data, img_dir, crop_size=(2944,1920)):
    """
    Function that loads a mammogram image using the meta data
    :param meta_data:
    :param img_dir:
    :param crop_size:
    :return:
    """
    img_path = os.path.join(img_dir, meta_data["hdf5_path"])
    loading_view = meta_data["View"][0] + "-" + meta_data["View"][1:]
    img = loading_mammogram.load_mammogram_img(img_path, crop_size, loading_view,
                                               meta_data["best_center"], meta_data["horizontal_flip"])
    img_pil = Image.fromarray(img / img.max())
    return img_pil


def collate_func_img(batch):
    """
    Collate functions used to collapse a mini-batch for single modal dataset.
    :param batch:
    :return:
    """
    crops_list = []
    batch_list = []
    cancer_label_list = []
    meta_list = []
    for i, (crops, cancer_label, meta_dict) in enumerate(batch):
        cancer_label_list.append(cancer_label.unsqueeze(0))
        meta_list.append(meta_dict)
        for j,img in enumerate(crops):
            if i==0:
                crops_list.append([img.unsqueeze(0)])
            else:
                crops_list[j].append(img.unsqueeze(0))

    for img_list in crops_list:
        batch_list.append(torch.cat(img_list,dim=0))

    return batch_list, torch.cat(cancer_label_list, dim=0), meta_list


def collate_func_breast(batch):
    flatten = list(itertools.chain(*batch))
    return collate_func_img(flatten)


@gin.configurable
def resolve_cancer_label(datum, cancer_label_col="image_cancer_label_mml"):
    label = datum[cancer_label_col]
    if cancer_label_col == "image_cancer_label_mml":
        if label == "n/a":
            return torch.Tensor([0,0])
        else:
            return torch.Tensor([label["benign"], label["malignant"]])
    elif cancer_label_col == "birads":
        birads_mapping = {0:0, 1:1, 2:1, 3:1, 4:2, 5:2}
        # missing birads
        if label == -1:
            if datum["image_cancer_label_mml"] == "n/a":
                return torch.Tensor([1]).long()
            elif datum["image_cancer_label_mml"]["malignant"] == 1:
                return torch.Tensor([2]).long()
            else:
                return torch.Tensor([1]).long()
        else:
            return torch.Tensor([birads_mapping[label]]).long()



class ImageDataset(Dataset):
    def __init__(self, data_list, img_dir, seg_dir, transformations, 
        check_positive_func=img_dl_pos_func, pos_to_neg_ratio=None,):
        self.pos_to_neg_ratio = pos_to_neg_ratio
        self.check_positive_func = check_positive_func
    
        self.data_list = data_list
        self.img_dir = img_dir
        self.seg_dir = seg_dir
        self.transformations = transformations

        self.positive_cases = [x for x in self.data_list if self.check_positive_func(x)]
        self.negative_cases = [x for x in self.data_list if not self.check_positive_func(x)]
        # Resample if requested
        if self.pos_to_neg_ratio is not None:
            self.resample()
        
    @staticmethod
    def load_single_image(data_pac, img_dir, seg_dir, transformations, index):
        # step #1: load pil images
        img_pil = load_mammogram_img(data_pac, img_dir)

        # step #2: load classification labels
        cancer_label = resolve_cancer_label(data_pac)
        img = transformations(img_pil)
        
        return img, cancer_label, data_pac
    
    def resample(self, pos_to_neg_ratio=None):
        """
        Resample self.data_list to include all positive samples + some randomly sampled negative samples.
        :param pos_to_neg_ratio:
        :return:
        """
        # Determine how many negative samples do we need.
        if pos_to_neg_ratio is None:
            pos_to_neg_ratio = self.pos_to_neg_ratio
        neg_need_num = np.minimum(int(round(len(self.positive_cases) * pos_to_neg_ratio)), len(self.negative_cases))
        random_idx = np.random.permutation(range(len(self.negative_cases)))[:neg_need_num]
        need_negative_cases = [self.negative_cases[idx] for idx in random_idx]
        self.data_list = self.positive_cases + need_negative_cases
        print(
        "After upsampling: {} positive exams {} negative exams".format(len(self.positive_cases), len(need_negative_cases)))
        # Reshuffle datalist.
        np.random.shuffle(self.data_list)

    def __getitem__(self, index):
        data_pac = self.data_list[index]
        return ImageDataset.load_single_image(data_pac, self.img_dir, self.seg_dir,
                                              self.transformations, index)

    def __len__(self):
        return len(self.data_list)


class UpsampleLoader:
    """
    A wrapper of dataset and dataloader which resamples data list for every epoch
    """
    def __init__(self, data_list, img_dir, seg_dir, transformations,
                 pos_to_neg_ratio, check_positive_func, dataset_constructor,
                 num_workers, collate_fn, batch_size=None, shuffle=False,
                 max_numel_per_batch=None, numel_col=None):
        # dataset args
        self.data_list = data_list
        self.img_dir = img_dir
        self.seg_dir = seg_dir
        self.transformations=transformations
        # sample selection args
        self.check_positive_func = check_positive_func
        self.pos_to_neg_ratio = pos_to_neg_ratio
        self.shuffle = shuffle
        # loading args
        self.num_workers = num_workers
        self.collate_fn = collate_fn
        self.batch_size = batch_size
        self.dataset_constructor = dataset_constructor
        self.max_numel_per_batch = max_numel_per_batch
        self.numel_col = numel_col
        self.resample()

    def resample(self):
        # resample the data list
        positive_cases = [x for x in self.data_list if self.check_positive_func(x)]
        negative_cases = [x for x in self.data_list if not self.check_positive_func(x)]
        neg_need_num = np.minimum(int(round(len(positive_cases) * self.pos_to_neg_ratio)), len(negative_cases))
        random_idx = np.random.permutation(range(len(negative_cases)))[:neg_need_num]
        need_negative_cases = [negative_cases[idx] for idx in random_idx]
        self.upsample_data_list=positive_cases+need_negative_cases
        print("After upsampling: {} positive exams {} negative exams".format(len(positive_cases), len(need_negative_cases)))
        np.random.shuffle(self.upsample_data_list)
        # create new dataset object
        self.dataset = self.dataset_constructor(self.upsample_data_list, self.img_dir, self.seg_dir, self.transformations)
        # create new data loader
        if self.max_numel_per_batch is not None:
            # create a batch sampler using the current data list
            assert self.numel_col is not None
            sampler = MaxImageNumberSampler(self.upsample_data_list, max_numel_per_batch=self.max_numel_per_batch, random=True,
                                                thresholds=range(2, self.max_numel_per_batch + 1), numel_col=self.numel_col)
            # create a data loader
            self.data_loader = DataLoader(self.dataset, num_workers=self.num_workers,
                                          collate_fn=self.collate_fn, pin_memory=True, batch_sampler=sampler)
        else:
            self.data_loader = DataLoader(self.dataset, num_workers=self.num_workers, batch_size=self.batch_size,
                                    collate_fn=self.collate_fn, pin_memory=True, shuffle=self.shuffle)

    def __iter__(self):
        for batch in self.data_loader:
            yield batch
        # reshuffle and re-sample the negative cases every epoch
        if self.shuffle:
            self.resample()

    def __len__(self):
        return len(self.upsample_data_list)


class BucketQueue:
    """
    Object that queues each exam according to number of images per exam
    """
    def __init__(self, data_list, max_numel_per_batch, thresholds=None, numel_col="num_imgs"):
        self.data_list = data_list
        self.max_numel_per_batch = max_numel_per_batch
        self.numel_col = numel_col
        numel = np.array([x.get(numel_col) for x in data_list])
        indices = np.arange(len(data_list))
        # create thresholds
        if thresholds is None:
            self.bucket_thresholds = [np.percentile(numel, 30), np.percentile(numel, 60),
                                      np.percentile(numel, 75), np.percentile(numel, 90),
                                      np.max(numel)]
        else:
            self.bucket_thresholds = thresholds
        # create list of queues
        self.idx_queue_list = [[] for _ in range(len(self.bucket_thresholds))]
        for i in range(len(numel)):
            for j in range(len(self.bucket_thresholds)):
                if numel[i] <= self.bucket_thresholds[j]:
                    self.idx_queue_list[j].append((indices[i], numel[i]))
                    break
        self.deplete = False


    def update(self):
        """
        Method that removes deleted queues and thresholds
        :return:
        """
        self.bucket_thresholds = [self.bucket_thresholds[i] for i in range(len(self.bucket_thresholds)) if len(self.idx_queue_list[i]) > 0]
        self.idx_queue_list = [self.idx_queue_list[i] for i in range(len(self.idx_queue_list)) if len(self.idx_queue_list[i]) > 0]
        if len(self.bucket_thresholds) == 0:
            self.deplete = True

    def sample_a_batch(self, random=False):
        """
        Method that samples a minibatch from the queue list
        :param random:
        :return:
        """
        output = []
        current_limit = self.max_numel_per_batch
        need_sample = True
        while need_sample:
            # select the bucket with largest number of elements
            largest_threshold_idx = bisect_left(self.bucket_thresholds, current_limit)
            selected_queue = self.idx_queue_list[largest_threshold_idx - 1]
            # select an element out of the bucket
            if random:
                bucket_idx = np.random.randint(low=0, high=len(selected_queue))
                current_data_idx, numel_added = selected_queue[bucket_idx]
                del selected_queue[bucket_idx]
            else:
                current_data_idx, numel_added = selected_queue.pop()
            # update status
            output.append(current_data_idx)
            current_limit -= numel_added
            self.update()
            # check if we can take more images
            if self.deplete or current_limit <= self.bucket_thresholds[0]:
                need_sample = False
        return output

    def give_all_batches(self, random=False):
        """
        Method that creates all minibatch index
        :param random:
        :return:
        """
        all_batches = []
        while not self.deplete:
            all_batches.append(self.sample_a_batch(random))
        return all_batches

@gin.configurable
class MaxImageNumberSampler(Sampler):
    """
    Object that creates minibatches which has strictly less number of images than the input
    """
    def __init__(self, data_list, max_numel_per_batch=6, random=False, thresholds=None, numel_col="num_imgs"):
        super(MaxImageNumberSampler).__init__()
        self.random = random
        self.thresholds = thresholds
        self.numel_col = numel_col
        self.data_list = data_list
        self.max_numel_per_batch = max_numel_per_batch
        self.recompute_batches(data_list, max_numel_per_batch, random)

    def recompute_batches(self, data_list, max_numel_per_batch, random):
        # create a queuelist object
        self.bucket_queue = BucketQueue(data_list, max_numel_per_batch,
                                        thresholds=self.thresholds, numel_col=self.numel_col)
        # calculate all batch index
        self.all_batches = self.bucket_queue.give_all_batches(random)

    def __iter__(self):
        if self.random:
            self.recompute_batches(self.data_list, self.max_numel_per_batch, True)
            np.random.shuffle(self.all_batches)
        for batch in self.all_batches:
            yield batch

    def __len__(self):
        return len(self.all_batches)
