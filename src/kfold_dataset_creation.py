import os
import random
import shutil
from sklearn.model_selection import KFold, train_test_split

input_dataset_dir = '../data/original_dataset/'
output_dataset_dir = '../data/kfold_dataset/'


def create_splits(k_folds=5):
    class_names = os.listdir(input_dataset_dir)

    # creating output directory
    os.makedirs(output_dataset_dir, exist_ok=True)

    # Perform K-Fold cross-validation
    kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)

    # Iterating over each class
    for class_name in class_names:
        class_path = os.path.join(input_dataset_dir, class_name)

        # Listing all images of that class
        all_images = os.listdir(class_path)
        random.shuffle(all_images)
        # Getting labels for each image
        labels = [class_name] * len(all_images)

        # Split the dataset into train and test sets using k-fold
        for fold, (_, _) in enumerate(kf.split(all_images, labels)):
            # Create fold directories
            train_fold_dir = os.path.join(output_dataset_dir, f'fold_{fold + 1}', 'train', class_name)
            val_fold_dir = os.path.join(output_dataset_dir, f'fold_{fold + 1}', 'validation', class_name)
            test_fold_dir = os.path.join(output_dataset_dir, f'fold_{fold + 1}', 'test', class_name)

            os.makedirs(train_fold_dir, exist_ok=True)
            os.makedirs(val_fold_dir, exist_ok=True)
            os.makedirs(test_fold_dir, exist_ok=True)

            print(f'Directories created...')

            print(f'Doing train-test-split...')
            train_images, val_test_images = train_test_split(all_images, test_size=0.4, stratify=labels,
                                                             random_state=42)
            print(f'Doing validation test split...')
            val_images, test_images = train_test_split(val_test_images, test_size=0.5,
                                                       stratify=[class_name] * len(val_test_images), random_state=42)

            print('Copying training images...')
            for image in train_images:
                shutil.copy(os.path.join(class_path, image), os.path.join(train_fold_dir, image))

            print('Copying validation images...')
            for image in val_images:
                shutil.copy(os.path.join(class_path, image), os.path.join(val_fold_dir, image))

            print('Copying test images...')
            for image in test_images:
                shutil.copy(os.path.join(class_path, image), os.path.join(test_fold_dir, image))


create_splits()
