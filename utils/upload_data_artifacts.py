import wandb
import os
from random import shuffle
import zipfile
import urllib.request as Request


def unzip_file(zip_src, dst_dir):
    r = zipfile.is_zipfile(zip_src)
    if r:
        fz = zipfile.ZipFile(zip_src, 'r')
        for file in fz.namelist():
            fz.extract(file, dst_dir)
    else:
        print('This is not zip')


def download_data(_save_path, _url):
    try:
        Request.urlretrieve(_url, _save_path)
        return True
    except:
        print('\nError when retrieving the URL:\n{}'.format(_url))
        return False


def create_artifacts(rootdir, data_dir, n_train=0.7, n_val=0.1, n_test=0.2, n_samples=27000, dataset_name='eurosat'):
    '''
    n_samples: the size of the dataset
    '''
    PROJECT_NAME = dataset_name
    RAW_DATA_AT = "_".join(["raw_data", str(n_samples)])
    SPLIT_DATA_AT = "_".join([f"split-{n_train}-{n_val}-{n_test}", str(n_samples)])
    SRC = rootdir + data_dir
    SPLIT_COUNTS = {
        "val": int(n_val * n_samples/10), #TODO: num_classes
        "test": int(n_test * n_samples/10),
        "train": int(n_train * n_samples),
    }
    if not os.path.exists(SRC):
        download_data(rootdir + '/data/EuroSAT_RGB.zip', 'http://madm.dfki.de/files/sentinel/EuroSAT.zip')
        unzip_file(rootdir + '/data/EuroSAT_RGB.zip', rootdir + '/data/EuroSAT_RGB')
    labels = sorted(os.listdir(SRC))  ##!!for reproducibility, sort them!
    run_split = wandb.init(project=PROJECT_NAME, job_type="data_split")
    # run_upload = wandb.init(project=PROJECT_NAME, job_type="upload")
    data_split_at = wandb.Artifact(SPLIT_DATA_AT, type="balanced_data")
    preview_dt = wandb.Table(columns=["id", "image", "label", "split"])
    # try:
    #     # raw data artifact exists
    #     raw_data_at = run_split.use_artifact(RAW_DATA_AT + ":latest")
    #     raw_data_exists = True
    # except:
    #     raw_data_at = wandb.Artifact(RAW_DATA_AT, type="raw_data")
    #     raw_data_exists = False

    for l in labels:
        imgs_per_label = os.path.join(SRC, l)
        if os.path.isdir(imgs_per_label):
            # filter out "DS_Store"
            imgs = [i for i in os.listdir(imgs_per_label) if not i.startswith(".DS")]
            # randomize the order
            shuffle(imgs)
            start_id = 0
            for split, count in SPLIT_COUNTS.items():
                split_imgs = imgs[start_id:start_id + count]
                for img_file in split_imgs:
                    f_id = img_file.split(".")[0]
                    full_path = os.path.join(SRC, l, img_file)
                    data_split_at.add_file(full_path, name=os.path.join(split, l, img_file))
                    preview_dt.add_data(f_id, wandb.Image(full_path), l, split)
                start_id += count


    # save artifact to W&B
    # if not raw_data_exists:
    #     run_upload.log_artifact(raw_data_at)
    data_split_at.add(preview_dt, "data_split")
    run_split.log_artifact(data_split_at)
    run_split.finish()
    # run_upload.finish()
                    # if not raw_data_exists:
                    #     raw_data_at.add_file(full_path, name=l + "/" + img_file)
                    # add a preview of the image