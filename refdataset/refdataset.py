import os
import numpy as np
import torch.utils.data as data
from PIL import Image
from torch.utils.data.dataloader import default_collate
from .refer import REFER 


class Refdataset(data.Dataset):
    def __init__(self,
                 refer_data_root='./data',
                 dataset='refcoco',
                 splitBy='unc',
                 split='train',
                 image_transforms=None,):
        """
        # Dataset loader supporting refcoco/+/g and the new JSON format datasets (using ref_id to obtain original information).
        
        Args:
            refer_data_root (str): Root directory for the REFER dataset.
            dataset (str): Dataset name ('refcoco', 'refcocog').
            splitBy (str): Split type ('unc', 'umd').
            split (str): Dataset split ('train', 'val', 'test').
            image_transforms (callable): Transformations for image and mask.
            new_dataset_json_path (str): JSON file path for the new dataset.
        """
        self.refer_data_root = refer_data_root
        self.dataset = dataset
        self.splitBy = splitBy
        self.image_transforms = image_transforms
        self.split = split

        if dataset == "arefcoco":
            data_dir = os.path.join(refer_data_root, dataset)
            new_dataset_json_path = os.path.join(data_dir, f'{dataset}.json')
            self._load_new_dataset(new_dataset_json_path)
        else:
            self._load_refer_dataset()

    def _load_new_dataset(self, json_path):
        """Load new dataset from JSON file."""
        import json
        print(f"Loading new dataset from {json_path} ...")

        with open(json_path, 'r') as f:
            data = json.load(f)

        self.refer = REFER(self.refer_data_root, "refcoco", "unc")
        self.referg = REFER(self.refer_data_root, "refcocog", "umd")
        self.annotations = [ann for ann in data['annotations'] if ann['split'] == self.split]
        self.entity_ids = [ann['entity_id'] for ann in self.annotations]

    def _load_refer_dataset(self):
        """Load original REFCOCO/+/G dataset."""
        print('Preparing REFER dataset ......')
        self.refer = REFER(self.refer_data_root, self.dataset, self.splitBy)
        self.ref_ids = self.refer.getRefIds(split=self.split)
        self.all_sentences = []
        self.all_category_id = []
        for index, x in enumerate(self.ref_ids):
            
            ref = self.refer.Refs[x]

            sentences_raw_for_ref = []

            for i,(sents,sent_id) in enumerate(zip(ref['sentences'],ref['sent_ids'])):
                sentence_raw = sents['sent']
                sentences_raw_for_ref.append(sentence_raw)
            
            self.all_sentences.append(sentences_raw_for_ref)

    def __len__(self):
        return len(self.entity_ids) if hasattr(self, 'entity_ids') else len(self.ref_ids)

    def __getitem__(self, index):
    
        if hasattr(self, 'annotations'):
            ann = self.annotations[index]
            ref_id = ann['ref_id']  
            descriptions = ann['descriptions']
            original_dataset = ann['original_dataset']

            ref = self.refer.Refs[ref_id] if original_dataset == 'refcoco' else self.referg.Refs[ref_id]
            img_id = self.refer.getImgIds(ref_id)[0] if original_dataset == 'refcoco' else self.referg.getImgIds(ref_id)[0]
            img_info = self.refer.Imgs[img_id] if original_dataset == 'refcoco' else self.referg.Imgs[img_id]

            img_path = os.path.join(self.refer.IMAGE_DIR, img_info['file_name']) if original_dataset == 'refcoco' else os.path.join(self.referg.IMAGE_DIR, img_info['file_name'])
            ref_mask = np.array(self.refer.getMask(ref)['mask']) if original_dataset == 'refcoco' else np.array(self.referg.getMask(ref)['mask'])
        
        else:
            ref_id = self.ref_ids[index]
            ref = self.refer.Refs[ref_id]
            img_id = self.refer.getImgIds(ref_id)[0]
            img_info = self.refer.Imgs[img_id]

            img_path = os.path.join(self.refer.IMAGE_DIR, img_info['file_name'])
            ref_mask = np.array(self.refer.getMask(ref)['mask'])

            descriptions = self.all_sentences[index]
        
        img = Image.open(img_path).convert("RGB")
        annot = np.zeros(ref_mask.shape)
        annot[ref_mask == 1] = 1
        org_gt = annot
        annot = Image.fromarray(annot.astype(np.uint8), mode='P')


        if self.image_transforms is not None:
            h, w = ref_mask.shape
            img, target = self.image_transforms(img, annot)
            target = target.unsqueeze(0)

        if self.split == 'train':
            chosen_description = np.random.choice(descriptions)
        else:
            chosen_description = descriptions # list of sentences

        batch = {
            'query_img': img,
            'query_mask': target,
            'query_idx': index,
            'sentence': chosen_description,
            'org_gt':org_gt,
        }

        return batch


def collate_fn(batch):
    batched_data = {}
    for key in batch[0].keys():
        if key == 'sentence' and isinstance(batch[0][key], list):
            batched_data[key] = [d[key] for d in batch]
        elif key == 'org_gt':
            batched_data[key] = [d[key] for d in batch]
        else:
            batched_data[key] = default_collate([d[key] for d in batch])

    return batched_data