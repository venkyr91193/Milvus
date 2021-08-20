import logging
from typing import List

from milvus import IndexType, MetricType, Milvus


# globals
MILVUS_SERVER_IP = "localhost"
MILVUS_SERVER_PORT = 19530
DIMENSION_OF_VECTORS = 1024  # DEPENDS ACCORDING TO MODELS USED
INDEX_FILE_SIZE = 1024  # CHUNKS OF DATA STORAGE SIZE IN MILVUS
NUMBER_OF_CLUSTERS = 4096  # NO OF CLUSTERS TO FORM INSIDE MILVUS

milvus_client = None

# milvus client
try:
    milvus_client = Milvus(
        host=MILVUS_SERVER_IP,
        port=MILVUS_SERVER_PORT,
    )
    if milvus_client == None:
        raise Exception("Milvus client object is None")
    logging.info(f"Initialized milvus client")
except Exception as e:
    logging.error(f"Error in initializing milvus client. {e}")

def add_vectors_to_milvus(
    collection_name: str,
    q_records: List[List[float]],
    node_ids: List[int],
    partition_tag: str = None,
) -> bool:
    """Function to add the vectors to milvus

    Args:
        collection_name (str): collection name of the database
        partition_tag (str): a partition inside the collection. Defaults to None.
        q_records (List[List[float]]): a list of list of floating numbers
        node_ids (List[int]): a list of node ids

    Raises:
        Exception: if not able to insert the vectors
        Exception: if not able to flush the data

    Returns:
        bool: True if no errors else False
    """
    try:
        if partition_tag:
            status, inserted_vector_ids = milvus_client.insert(
                collection_name=collection_name,
                partition_tag=partition_tag,
                records=q_records,
                ids=node_ids,
            )
            if status.code != 0:
                raise Exception(status.message)
            logging.info(
                f"Inserted vector into collection_name={collection_name}, partition_tag={partition_tag}, vector_ids={node_ids}"
            )
        else:
            status, inserted_vector_ids = milvus_client.insert(
                collection_name=collection_name,
                records=q_records,
                ids=node_ids,
            )
            if status.code != 0:
                raise Exception(status.message)
            logging.info(
                f"Inserted vector into collection_name={collection_name}, vector_ids={node_ids}"
            )
        status = milvus_client.flush(collection_name_array=[collection_name])
        if status.code != 0:
            raise Exception(status.message)
        logging.info(f"Flushed data for collection_name={collection_name}")
    except Exception as e:
        logging.error(e)
        return False
    return True


def delete_vectors_from_milvus(
    collection_name: str, node_ids: List[int]
) -> bool:
    """Function to delete vectors from milvus

    Args:
        collection_name (str): collection name of the database
        node_ids (List[int]): list of node ids corresponding to the vectors

    Raises:
        Exception: if not able to delete the vectors
        Exception: if not able to flush the data
        Exception: if not able to free up space in milvus after deletion

    Returns:
        bool: True if no errors else False
    """
    if not len(node_ids):
        return True

    try:
        # deleting the nodes
        status = milvus_client.delete_entity_by_id(
            collection_name=collection_name, id_array=node_ids
        )
        if status.code != 0:
            raise Exception(status.message)
        logging.info(
            f"Deleted vectors from collection_name={collection_name}, vector_ids={node_ids}"
        )
        # freeing up space
        status = milvus_client.compact(collection_name=collection_name)
        if status.code != 0:
            raise Exception(status.message)
        status = milvus_client.flush(collection_name_array=[collection_name])
        if status.code != 0:
            raise Exception(status.message)
        logging.info(f"Flushed data for collection_name={collection_name}")
    except Exception as e:
        logging.error(e)
        return False
    return True


def query_milvus(
    collection_name: str,
    q_records: List[str],
    partition_tag: str = None,
    tok_k: int = 10,
) -> list:
    """Function to query milvus

    Args:
        collection_name (str): collection name of the database
        q_records (List[List[float]]): the input records to be queried
        partition_tag (str): a partition inside the collection. Defaults to None.
        tok_k (int): number of top results to be returned. Default to 10.

    Raises:
        Exception: if not able to query milvus

    Returns:
        list: a list of results for multiple queries
    """
    results = list()
    try:
        search_param = {"nprobe": 128}
        if partition_tag:
            # change the number of top results to store
            status, results = milvus_client.search(
                collection_name=collection_name,
                partition_tags=[partition_tag],
                query_records=q_records,
                top_k=tok_k,
                params=search_param,
            )
        else:
            # change the number of top results to store
            status, results = milvus_client.search(
                collection_name=collection_name,
                query_records=q_records,
                top_k=tok_k,
                params=search_param,
            )
        if status.code != 0:
            raise Exception(status.message)
        logging.info(
            f"Queried collection_name={collection_name}, partition_tag={partition_tag}"
        )
    except Exception as e:
        logging.error(e)
    return results


def check_and_add_collection(
    collection_name: str, dimension_of_vectors: int = DIMENSION_OF_VECTORS
) -> bool:
    """Function to check if a collection exists, else to create it

    Args:
        collection_name (str): collection name of the database
        dimension_of_vectors (int): dimension of the vectors used

    Raises:
        Exception: if not able to check for collection
        Exception: if not able to create index for new collection
        Exception: if not able to create the new collection

    Returns:
        bool: True if no errors else False
    """
    try:
        status, ok = milvus_client.has_collection(collection_name=collection_name)
        if status.code != 0:
            raise Exception(status.message)
        if not ok:
            param = {
                "collection_name": collection_name,
                "dimension": dimension_of_vectors,
                "index_file_size": INDEX_FILE_SIZE,
                "metric_type": MetricType.IP,
            }
            status = milvus_client.create_collection(param)
            if status.code != 0:
                raise Exception(status.message)
            logging.info(f"Created collection_name={collection_name}")
            # create index
            ivf_param = {"nlist": NUMBER_OF_CLUSTERS}
            status = milvus_client.create_index(
                collection_name, IndexType.IVF_SQ8, ivf_param
            )
            if status.code != 0:
                raise Exception(status.message)
            logging.info(f"Created index for collection_name={collection_name}")
    except Exception as e:
        logging.error(e)
        return False
    return True


def check_for_collection(collection_name: str) -> bool:
    """Function to check if a collection exists

    Args:
        collection_name (str): collection name of the database

    Raises:
        Exception: if not able to check for collection

    Returns:
        bool: True if collection exists else False
    """
    try:
        status, ok = milvus_client.has_collection(collection_name=collection_name)
        if status.code != 0:
            raise Exception(status.message)
        if not ok:
            return False
        return True
    except Exception as e:
        logging.error(e)


def delete_collection(collection_name: str):
    """Function to delete a collection if exists

    Args:
        collection_name (str): collection name of the database

    Raises:
        Exception: if not able to check for collection
        Exception: if not able to delete the collection

    Returns:
        None
    """
    try:
        status, ok = milvus_client.has_collection(collection_name=collection_name)
        if status.code != 0:
            raise Exception(status.message)
        if not ok:
            logging.info(f"Invalid collection_name={collection_name}")
        else:
            status = milvus_client.drop_collection(collection_name=collection_name)
            if status.code != 0:
                raise Exception(status.message)
            logging.info(f"Deleted collection_name={collection_name}")
    except Exception as e:
        logging.error(e)


def check_and_add_partition(collection_name: str, partition_tag: str) -> bool:
    """Function to check if the parition exists, if not create it

    Args:
        collection_name (str): collection name of the database
        partition_tag (str): a partition inside the collection

    Raises:
        Exception: if milvus is not able to list the partitions
        Exception: if milvus is not able to create a partition

    Returns:
        bool: True if no errors else False
    """
    try:
        status, partitions = milvus_client.list_partitions(
            collection_name=collection_name
        )
        if status.code != 0:
            raise Exception(status.message)
        if partition_tag not in [partition.tag for partition in partitions]:
            status = milvus_client.create_partition(
                collection_name=collection_name,
                partition_tag=partition_tag,
            )
            if status.code != 0:
                raise Exception(status.message)
            logging.info(
                f"Created partition for collection_name={collection_name}, partition_tag={partition_tag}"
            )
    except Exception as e:
        logging.error(e)
        return False
    return True


def get_milvus_node_list(collection_name: str) -> List[int]:
    """Function to get the list of node_ids from milvus

    Args:
        collection_name (str): collection name of the database

    Raises:
        Exception: if not able to get collection stats

    Returns:
        List[int]: a list of node_ids
    """
    # getting nodes from milvus
    milvus_nodes = list()
    try:
        status, collection_info = milvus_client.get_collection_stats(
            collection_name=collection_name
        )
        if status.code != 0:
            raise Exception(status.message)
        partition_list = collection_info["partitions"]
        for partition in partition_list:
            segment_list = partition["segments"]
            if segment_list is not None:
                for segment in segment_list:
                    segment_name = segment["name"]
                    status, id_list = milvus_client.list_id_in_segment(
                        collection_name=collection_name, segment_name=segment_name
                    )
                    milvus_nodes.extend(id_list)
                    if status.code != 0:
                        logging.warning(status.message)
    except Exception as e:
        logging.warning(e)
    return milvus_nodes


def count_total_vectors(collection_list: List[str] = []) -> dict:
    """Function to return the count of the vectors in milvus

    Args:
        collection_list (List(str)): collection names of the database. Defaults to []

    Raises:
        Exception: if not able to count the vectors in the collection

    Returns:
        dict: dict of keys material and subject with respective count
    """
    total_count = 0
    try:
        if not len(collection_list):
            status, collection_list = milvus_client.list_collections()
            if status.code != 0:
                raise Exception(status.message)

        for collection_name in collection_list:
            # material count
            try:
                status, ok = milvus_client.has_collection(
                    collection_name=collection_name
                )
                if ok:
                    status, count = milvus_client.count_entities(
                        collection_name=collection_name
                    )
                    if status.code != 0:
                        raise Exception(status.message)
                    if count != None:
                        total_count += count
            except Exception as e:
                logging.error(e)
    except Exception as e:
        logging.error(e)
    return {"total_count": total_count}
