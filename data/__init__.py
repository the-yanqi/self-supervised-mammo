import gin, pickle
from data import datasets, transformations
from torch.utils.data import DataLoader

@gin.configurable
def get_image_dataset(datalist_dir,
                      img_dir,
                      segmentation_dir,
                      num_workers,
                      data_mode,
                      max_img_per_batch=None,
                      batch_size=None,
                      resize=None,
                      pos_to_neg_ratio=None,
                      undersample=None,
                      imaging_modality="mammo",
                      image_format="greyscale",
                      training_augmentation="standard"):
    """
    Entry-level function
    :return:
    """
    # step #1: load data list
    with open(datalist_dir, "rb") as f:
        train_dl, val_dl, ts_dl, _ = pickle.load(f)

    # step #2: image transformations
    training_transformations = transformations.compose_transform(augmentation=training_augmentation, resize=resize, image_format=image_format)
    val_transformations = transformations.compose_transform(augmentation=None, resize=resize, image_format=image_format)
    test_transformations = transformations.compose_transform(augmentation=None, resize=resize, image_format=image_format)

    # step #3: create datasets
    if data_mode == "image":
        assert batch_size is not None
        tr_ds = datasets.ImageDataset(data_list=train_dl,
                                        img_dir=img_dir,
                                        seg_dir=segmentation_dir,
                                        imaging_modality=imaging_modality,
                                        transformations=training_transformations,
                                        pos_to_neg_ratio=pos_to_neg_ratio,
                                        purge=True)
        val_ds = datasets.ImageDataset(val_dl, img_dir, segmentation_dir, imaging_modality, val_transformations, purge=False)
        ts_ds = datasets.ImageDataset(ts_dl, img_dir, segmentation_dir, imaging_modality, test_transformations, purge=False)
    elif data_mode == "breast":
        assert max_img_per_batch is not None
        # TODO: this way the images threw out stay the same throughout the experiment
        # TODO: need to think about a way to make images threw out changed for each epoch
        train_dl = datasets.BreastDataset.group_dl_for_breast(train_dl)
        val_dl = datasets.BreastDataset.group_dl_for_breast(val_dl)
        ts_dl = datasets.BreastDataset.group_dl_for_breast(ts_dl)
        if undersample is not None:
            train_dl = [datasets.BreastDataset.undersample(x, undersample) for x in train_dl]
            val_dl = [datasets.BreastDataset.undersample(x, undersample) for x in val_dl]
            ts_dl = [datasets.BreastDataset.undersample(x, undersample) for x in ts_dl]
        tr_ds = datasets.BreastDataset(train_dl, img_dir, segmentation_dir, training_transformations)
        val_ds = datasets.BreastDataset(val_dl, img_dir, segmentation_dir, val_transformations)
        ts_ds = datasets.BreastDataset(ts_dl, img_dir, segmentation_dir, test_transformations)

    # step #4: create data loader
    if data_mode == "image":
        if pos_to_neg_ratio is None:
            tr_dld = DataLoader(tr_ds, batch_size, shuffle=True, num_workers=num_workers, collate_fn=datasets.collate_func_img, pin_memory=True)
        else:
            tr_dld = datasets.UpsampleLoader(dataset=tr_ds,  num_workers=num_workers, collate_fn=datasets.collate_func_img,
                                             batch_size=batch_size, shuffle=True)
        val_dld = DataLoader(val_ds, batch_size, shuffle=False, num_workers=num_workers, collate_fn=datasets.collate_func_img,
                             pin_memory=True)
        ts_dld = DataLoader(ts_ds, batch_size, shuffle=False, num_workers=num_workers, collate_fn=datasets.collate_func_img,
                            pin_memory=True)
    elif data_mode == "breast":
        # create sampler
        tr_sampler = datasets.MaxImageNumberSampler(train_dl, max_numel_per_batch=max_img_per_batch, random=True,
                                                    thresholds=range(2, max_img_per_batch + 1), numel_col="num_imgs")
        val_sampler = datasets.MaxImageNumberSampler(val_dl, max_numel_per_batch=max_img_per_batch, random=False,
                                                    thresholds=range(2, max_img_per_batch + 1), numel_col="num_imgs")
        ts_sampler = datasets.MaxImageNumberSampler(ts_dl, max_numel_per_batch=max_img_per_batch, random=False,
                                                    thresholds=range(2, max_img_per_batch + 1), numel_col="num_imgs")
        # create data loader
        if pos_to_neg_ratio is None:
            tr_dld = DataLoader(tr_ds, sampler=tr_sampler, num_workers=num_workers,
                                collate_fn=datasets.collate_func_breast, pin_memory=True)
        else:
            tr_dld = datasets.UpsampleLoader(dataset=tr_ds, num_workers=num_workers, collate_fn=datasets.collate_func_breast,
                                             max_numel_per_batch=max_img_per_batch, numel_col="num_imgs", shuffle=True)
        val_dld = DataLoader(val_ds, batch_sampler=val_sampler, num_workers=num_workers, collate_fn=datasets.collate_func_breast, pin_memory=True)
        ts_dld = DataLoader(ts_ds, batch_sampler=ts_sampler, num_workers=num_workers, collate_fn=datasets.collate_func_breast, pin_memory=True)
    return tr_dld, val_dld, ts_dld