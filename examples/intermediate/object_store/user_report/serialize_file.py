import pickle
import sys

from aiq.object_store.models import ObjectStoreItem

# Usage: python serialize_file.py <file_path>

if len(sys.argv) != 2:
    print("Usage: python serialize_file.py <file_path>")
    sys.exit(1)

file_path = sys.argv[1]

with open(file_path, "rb") as f:
    data = f.read()

item = ObjectStoreItem(data=data, )

with open(file_path + ".pkl", "wb") as f:
    pickle.dump(item, f)
