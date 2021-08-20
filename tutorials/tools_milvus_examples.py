import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__))))


from tools_milvus import *

# milvus should be running for these tests
def get_total_count():
    # total count
    count = count_total_vectors()
    assert(count == {'total_count': 0})

def add_delete_vector():
    # dimension should be 1024, can be changed in the tools_milvus file according to the model used
    vector = [1] * 1024
    assert(len(vector) == 1024)
    check_and_add_collection('example_collection', 1024)
    add_vectors_to_milvus('example_collection', [vector], [1])
    count = count_total_vectors()
    assert(count == {'total_count': 1})
    delete_vectors_from_milvus('example_collection', [1])
    count = count_total_vectors()
    assert(count == {'total_count': 0})

if __name__ == "__main__":
    get_total_count()
    add_delete_vector()
    print("All tests passed")