"""
Generate train.json in coco format from mask images.
Generate test.json
"""

# Imports
from pycococreatortools import *
from shapes_to_coco import *
from tqdm import tqdm


# Training and testing dataset directory path
IMAGE_DIR = '../dataset-small2/train'  # DBG
TEST_DIR = '../dataset/test'


# Get train images and mask images path
def get_train_dataset_path():
    train_imgs_path = []
    train_masks_path = []
    for root, folders, files in os.walk(IMAGE_DIR):
        image_files = filter_for_jpeg(root, files)
        if(len(image_files) == 1):
            train_imgs_path.append(image_files)
        elif(len(image_files) > 1):
            train_masks_path.append(image_files)

    # Delete a checkpoint image found in nowhere
    for img_path in train_imgs_path:
        if img_path[0].find('checkpoints') != -1:
            train_imgs_path.remove(img_path)

    # Check if the two path list are matched
    for i in range(len(train_imgs_path)):
        if(train_imgs_path[i][0][:40] != train_masks_path[i][0][:40]):
            raise Exception

    return train_imgs_path, train_masks_path


# Get train.json in coco format
def get_coco_train():
    train_imgs_path, train_masks_path = get_train_dataset_path()
    coco_train = {"images": [], "annotations": [], "categories": [{"id": 1, "name": "nuclei"}]}
    # go through each image
    image_id = 1
    annotation_id = 1
    for idx, image_path in tqdm(enumerate(train_imgs_path)):
        image_path = image_path[0]
        image = Image.open(image_path)
        image_info = create_image_info(
            image_id, os.path.basename(image_path), image.size)
        coco_train["images"].append(image_info)

        # filter for associated png annotations
        for mask_img in tqdm(train_masks_path[idx]):
            category_info = {'id': 1, 'is_crowd':0}
            binary_mask = np.asarray(Image.open(mask_img)
                .convert('1')).astype(np.uint8)
            
            annotation_info = create_annotation_info(
                annotation_id, image_id, category_info, binary_mask,
                image.size, tolerance=2)
            if annotation_info is not None:
                coco_train["annotations"].append(annotation_info)
            annotation_id += 1
        image_id += 1

    return coco_train


def get_coco_test():
    test_images_path = []
    for root, _, files in os.walk(TEST_DIR):
        image_files = filter_for_jpeg(root, files)
        test_images_path.append(image_files)

    # Delete a checkpoint image found in nowhere
    for img_path in test_images_path:
        if img_path[0].find('checkpoints') != -1:
            test_images_path.remove(img_path)   
   
    image_id = 1
    coco_test = {
      "images": [],
      "annotations": [],
      "categories": [{"id": 1, "name": "nuclei"}]
    }

    for image_path in test_images_path[0]:
        image = Image.open(image_path)
        image_info = create_image_info(
                image_id, os.path.basename(image_path), image.size)
        coco_test["images"].append(image_info)
        image_id += 1

    return coco_test


# Get train.json
coco_train = get_coco_train()
#json_train_file = '../train.json' # DBG to be restored
json_train_file = '../train_testcode.json' # DBG to be deleted
os.makedirs(os.path.dirname(json_train_file), exist_ok=True)
json_fp = open(json_train_file, "w")
json_str = json.dumps(coco_train, indent=4)
json_fp.write(json_str)
json_fp.close()


# Get test.json
coco_test = get_coco_test()
json_test_file = '../test.json'
os.makedirs(os.path.dirname(json_test_file), exist_ok=True)
json_fp = open(json_test_file, "w")
json_str = json.dumps(coco_test, indent=4)
json_fp.write(json_str)
json_fp.close()
