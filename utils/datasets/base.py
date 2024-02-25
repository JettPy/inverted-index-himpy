import os
import shutil
import urllib.request


"""
Common functions
"""


def _download(url, path):

    def show_download_progress(num, size, total):
        num_total = total / size
        progress = int(num / num_total * 100)
        print("\r[{:100}]".format("=" * progress + (">" if progress != 100 else "")), end="")

    filename = os.path.basename(url)
    archive_path = os.path.join(path, filename)
    tmp_filename = "~" + filename
    tmp_archive_path = os.path.join(path, tmp_filename)
    print("Downloading {}...".format(filename))
    urllib.request.urlretrieve(url, tmp_archive_path, reporthook=show_download_progress)
    os.rename(tmp_archive_path, archive_path)
    print("\nCompleted.")
    return archive_path


def _unpack(src_path, dst_path, delete_archive=True):
    shutil.unpack_archive(src_path, dst_path)
    if delete_archive is True:
        os.unlink(src_path)
