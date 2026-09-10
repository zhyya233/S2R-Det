# S²R-Det VisDrone2019-DET dataset fragment.
# COCO JSON files are generated outside Git under the dataset root.

dataset_type = 'CocoDataset'
data_root = 'data/links/visdrone2019_det/'

metainfo = dict(
    classes=(
        'pedestrian',
        'people',
        'bicycle',
        'car',
        'van',
        'truck',
        'tricycle',
        'awning-tricycle',
        'bus',
        'motor',
    )
)

train_dataset = dict(
    type=dataset_type,
    data_root=data_root,
    ann_file='coco_annotations/visdrone2019_det_train.json',
    data_prefix=dict(img='VisDrone2019-DET-train/images/'),
    metainfo=metainfo,
)

val_dataset = dict(
    type=dataset_type,
    data_root=data_root,
    ann_file='coco_annotations/visdrone2019_det_val.json',
    data_prefix=dict(img='VisDrone2019-DET-val/images/'),
    metainfo=metainfo,
)

testdev_dataset = dict(
    type=dataset_type,
    data_root=data_root,
    ann_file='coco_annotations/visdrone2019_det_testdev.json',
    data_prefix=dict(img='VisDrone2019-DET-test-dev/images/'),
    metainfo=metainfo,
)
