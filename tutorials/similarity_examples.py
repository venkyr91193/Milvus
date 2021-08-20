import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__))))


from tools_milvus import *
from model import Model

model_obj = Model('facebook/bart-large-mnli')

def modelling_example():
    sentences_index = {
        0: 'we are having problems with the model',
        1: 'the model is working very well',
        2: 'the sun is very hot today',
        3: 'today is a cloudy day',
    }
    ids = list(sentences_index.keys())
    sentences = list(sentences_index.values())
    # dimension should be 1024, can be changed in the tools_milvus file according to the model used
    vectors = model_obj.make_sentence_vectors(sentences)
    check_and_add_collection('example_collection', 1024)
    add_vectors_to_milvus('example_collection', vectors, ids)

    # query the most similar sentence
    new_sentence = "it is too sweaty today"
    new_vector = model_obj.make_sentence_vectors([new_sentence])
    result = query_milvus('example_collection', new_vector)
    for index,res in enumerate(result.id_array[0]):
        print(f"Sentence '{sentences_index.get(res)}' is similar by {result.distance_array[0][index]}")
    
    # delete the collection when existing if you want to run the code again with different examples
    delete_collection('example_collection')
    
if __name__ == "__main__":
    modelling_example()
    print("All tests passed")