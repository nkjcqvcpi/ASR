# encoding: utf-8
import os


class BaseImageDataset:
    """
    Base class of image reid dataset
    """
    """
        Base class of reid dataset
        """

    @staticmethod
    def get_imagedata_info(data):
        pids, cams, tracks = [], [], []

        for _, pid, camid, trackid in data:
            pids += [pid]
            cams += [camid]
            tracks += [trackid]
        pids = set(pids)
        cams = set(cams)
        tracks = set(tracks)
        num_pids = len(pids)
        num_cams = len(cams)
        num_imgs = len(data)
        num_views = len(tracks)
        return num_pids, num_imgs, num_cams, num_views

    def _check_before_run(self):
        """ Check if all files are available before going deeper """
        if not os.path.exists(self.dataset_dir):
            raise RuntimeError("'{}' is not available".format(self.dataset_dir))
        if not os.path.exists(self.train_dir):
            raise RuntimeError("'{}' is not available".format(self.train_dir))
        if not os.path.exists(self.query_dir):
            raise RuntimeError("'{}' is not available".format(self.query_dir))
        if not os.path.exists(self.gallery_dir):
            raise RuntimeError("'{}' is not available".format(self.gallery_dir))

    @classmethod
    def get_dataset_info(cls, train=None, query=None, gallery=None):
        if train:
            cls.num_train_pids, cls.num_train_imgs, cls.num_train_cams, cls.num_train_views = cls.get_imagedata_info(train)
        if query:
            cls.num_query_pids, cls.num_query_imgs, cls.num_query_cams, cls.num_query_views = cls.get_imagedata_info(query)
        if gallery:
            cls.num_gallery_pids, cls.num_gallery_imgs, cls.num_gallery_cams, cls.num_gallery_views = cls.get_imagedata_info(gallery)

    def print_dataset_statistics(self):
        print("Dataset statistics:")
        print("  ----------------------------------------")
        print("  subset  | # ids | # images | # roles")
        print("  ----------------------------------------")
        print("  train   | {:5d} | {:8d} | {:9d}".format(self.num_train_pids, self.num_train_imgs, self.num_train_views))
        print("  query   | {:5d} | {:8d} | {:9d}".format(self.num_query_pids, self.num_query_imgs, self.num_query_views))
        print("  gallery | {:5d} | {:8d} | {:9d}".format(self.num_gallery_pids, self.num_gallery_imgs, self.num_gallery_views))
        print("  ----------------------------------------")
