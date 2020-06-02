import hashlib
import os

DATA_RAW_PATH = "data/raw/yahoo_answers_csv.tar.gz"
DATA_RAW_MD5SUM = "f3f9899b997a42beb24157e62e3eea8d"

# Compare md5sum to avoid data corruption
# TODO : update with latest dataset
def test_dataset_integrity():
    # TODO: replace with chunk-based hash to reduce memory requirements.
    def checksum(file):
        def file_as_bytes(file):
            with file:
                return file.read()
        # Return md5 hash from file fully loaded in memory
        return hashlib.md5( file_as_bytes( open(file, 'rb') ) ).hexdigest()
    # Assert the datafile exists
    assert os.path.exists(DATA_RAW_PATH)
    # Compare current datahash to reference hash to avoid data corruption
    assert DATA_RAW_MD5SUM == checksum(DATA_RAW_PATH)
