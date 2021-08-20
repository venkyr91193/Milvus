#!/bin/sh

envsubst '$${MYSQL_USER},$${MYSQL_PASSWORD},$${MILVUS_CACHE_SIZE},$${MILVUS_INSERT_BUFFER_SIZE},$${MYSQL_DATABASE}' < /var/lib/milvus/server_config_template.yaml > /var/lib/milvus/conf/server_config.yaml

/var/lib/milvus/bin/milvus_server -c /var/lib/milvus/conf/server_config.yaml
