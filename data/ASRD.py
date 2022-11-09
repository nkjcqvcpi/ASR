# encoding: utf-8

import os
import os.path as osp

from .bases import BaseImageDataset


class ASRD(BaseImageDataset):
    dataset_dir = 'ASRD'
    code = {'US': '840', 'GB': '826', 'FR': '250', 'KR': '410', 'CA': '124', 'FI': '246', 'DE': '276', 'CS': '203',
            'JP': '392', 'CN': '156', 'TW': '158', 'HK': '344', 'PL': '616'}

    def __init__(self, root='./toDataset'):
        super(ASRD, self).__init__()
        self.dataset_dir = osp.join(root, self.dataset_dir)
        self.train_dir = osp.join(self.dataset_dir, 'train')
        self.query_dir = osp.join(self.dataset_dir, 'query')
        self.gallery_dir = osp.join(self.dataset_dir, 'test')

        self._check_before_run()

        self.train = self._process_dir(self.train_dir, relabel=True)
        self.query = self._process_dir(self.query_dir, relabel=False)
        self.gallery = self._process_dir(self.gallery_dir, relabel=False, mode=True)

        self.get_dataset_info(self.train, self.query, self.gallery)
        self.num_train_cams = self.num_train_vids = 0

        print("=> AniChFace loaded")
        self.print_dataset_statistics()

    def _process_dir(self, dir_path, relabel=True, mode=False):
        work_paths = [i for i in os.listdir(dir_path) if os.path.isdir(os.path.join(dir_path, i))]
        if relabel:
            pid_container = set(work_path for work_path in work_paths if os.path.isdir(os.path.join(dir_path, work_path)) and os.listdir(os.path.join(dir_path, work_path)))
            self.pid2label = {pid: label for label, pid in enumerate(pid_container)}
        dataset = []
        for work_path in work_paths:
            t = os.path.join(dir_path, work_path)
            char_paths = [i for i in os.listdir(t) if os.path.isdir(os.path.join(t, i))]
            for char_path in char_paths:
                t = os.path.join(dir_path, work_path, char_path)
                img_paths = [i for i in os.listdir(t) if i[0] != '.']
                for img_path in img_paths:
                    wid, cid, _ = img_path.split('_')
                    if relabel:
                        wid = self.pid2label[wid]
                    else:
                        wid = int(self.code[wid[:2]] + wid[2:])
                    cid = 0 if mode else 1
                    dataset.append((os.path.join(t, img_path), int(wid), cid, 0))  # int(cid)
        return dataset


class KCrossASRD(BaseImageDataset):
    dataset_dir = 'ASRDf'
    code = {'US': '840', 'GB': '826', 'FR': '250', 'KR': '410', 'CA': '124', 'FI': '246', 'DE': '276', 'CS': '203',
            'JP': '392', 'CN': '156', 'TW': '158', 'HK': '344', 'PL': '616'}

    def __init__(self, k, root='./toDataset'):
        super(KCrossASRD, self).__init__()
        dataset_dir = osp.join(root, self.dataset_dir)
        train_dirs = [osp.join(dataset_dir, d) for d in os.listdir(dataset_dir) if d != str(k) and d != '.DS_Store']
        test_dir = osp.join(dataset_dir, str(k))

        self.train = []
        pid_container = set()
        for dr in train_dirs:
            for d in [d for d in os.listdir(dr) if d != '.DS_Store']:
                pid_container.add(d)
        self.pid2label = {pid: label for label, pid in enumerate(pid_container)}

        for dr in train_dirs:
            self._process_train_dir(dr)

        self.query = []
        self.gallery = []
        self._process_test_dir(test_dir)

        self.get_dataset_info(self.train, self.query, self.gallery)
        print("=> AniChFace loaded")
        self.print_dataset_statistics()
        self.num_train_cams = self.num_train_vids = 0

    def _process_train_dir(self, dir_path):
        for img_paths in os.listdir(dir_path):
            path = os.path.join(dir_path, img_paths)
            if os.path.isdir(path):
                for character in os.listdir(path):
                    cp = os.path.join(path, character)
                    if os.path.isdir(cp):
                        for img_path in [i for i in os.listdir(cp) if i[0] != '.' and osp.isfile(osp.join(cp, i))]:
                            wid, _, _ = img_path.split('_')
                            wid = self.pid2label[wid]
                            self.train.append((os.path.join(path, character, img_path), int(wid), 0, 0))

    def _process_test_dir(self, dir_path):
        for img_paths in os.listdir(dir_path):
            path = os.path.join(dir_path, img_paths)
            if os.path.isdir(path):
                characters = sorted([int(i) for i in os.listdir(path) if os.path.isdir(os.path.join(path, i))])
                query = characters[:max(int(len(characters) * 0.4), 1)]
                gallery = characters[max(int(len(characters) * 0.4), 1):]
                for character in characters:
                    c_path = os.path.join(path, '{:04}'.format(character))
                    for img_path in os.listdir(c_path):
                        if img_path[0] != '.' and os.path.isfile(os.path.join(c_path, img_path)):
                            wid, cid, _ = img_path.split('_')
                            wid = self.code[wid[:2]] + wid[2:]
                            im_path = osp.join(path, '{:04}'.format(character), img_path)
                            if int(cid) in query:
                                self.query.append((im_path, int(wid), 1, int(wid + cid)))
                            elif int(cid) in gallery:
                                self.gallery.append((im_path, int(wid), 0, int(wid + cid)))


if __name__ == '__main__':
    ds = [KCrossASRD(i, '/Users/houtonglei/Documents/') for i in range(5)]
    i = 0