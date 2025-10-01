#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Test script for SQL Retriever using MySQL and Elasticsearch.

This script demonstrates how to use the ElasticNIMVanna class with MySQL database
and Elasticsearch for vector storage.

Prerequisites:
    - MySQL server running with accessible database
    - Elasticsearch running (default: http://localhost:9200)
    - NVIDIA_API_KEY environment variable set
    - Required packages: pymysql, elasticsearch, vanna

Usage:
    # Set your API key
    export NVIDIA_API_KEY="nvapi-your-key-here"
    
    # Run with default question
    python test_mysql_elastic.py
    
    # Run with custom question
    python test_mysql_elastic.py -q "What are the top 5 customers by revenue?"
    
    # Specify custom MySQL connection
    python test_mysql_elastic.py --mysql-host localhost --mysql-user myuser --mysql-password mypass --mysql-database mydb
    
    # Specify custom Elasticsearch connection
    python test_mysql_elastic.py --es-url http://localhost:9200 --es-index my_sql_vectors
"""

import argparse
import logging
import os
import sys

from vanna_util import ElasticNIMVanna
from vanna_util import NVIDIAEmbeddingFunction
from vanna_util import init_vanna

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def check_elasticsearch_connection(es_url: str) -> bool:
    """
    Check if Elasticsearch is accessible.
    
    Args:
        es_url: Elasticsearch URL
        
    Returns:
        True if connection successful
    """
    try:
        from elasticsearch import Elasticsearch
        
        es_client = Elasticsearch(es_url)
        if es_client.ping():
            logger.info(f"✓ Elasticsearch is accessible at {es_url}")
            return True
        else:
            logger.error(f"✗ Cannot connect to Elasticsearch at {es_url}")
            return False
    except ImportError:
        logger.error("✗ elasticsearch package not installed. Run: pip install elasticsearch")
        return False
    except Exception as e:
        logger.error(f"✗ Error connecting to Elasticsearch: {e}")
        return False


def check_mysql_connection(host: str, user: str, password: str, database: str, port: int = 3306) -> bool:
    """
    Check if MySQL server is accessible.
    
    Args:
        host: MySQL host
        user: MySQL username
        password: MySQL password
        database: MySQL database name
        port: MySQL port (default: 3306)
        
    Returns:
        True if connection successful
    """
    try:
        import pymysql
        
        connection = pymysql.connect(
            host=host,
            user=user,
            password=password,
            database=database,
            port=port
        )
        connection.close()
        logger.info(f"✓ MySQL is accessible at {host}:{port}/{database}")
        return True
    except ImportError:
        logger.error("✗ pymysql package not installed. Run: pip install pymysql")
        return False
    except Exception as e:
        logger.error(f"✗ Error connecting to MySQL: {e}")
        return False


def needs_training(es_client, index_name: str) -> bool:
    """
    Check if the Elasticsearch index needs training by checking if it has documents.
    
    Args:
        es_client: Elasticsearch client
        index_name: Name of the index
        
    Returns:
        True if training is needed
    """
    try:
        if not es_client.indices.exists(index=index_name):
            logger.info("Index does not exist -> needs training")
            return True
        
        # Check document count
        count_response = es_client.count(index=index_name)
        doc_count = count_response.get("count", 0)
        
        if doc_count == 0:
            logger.info("Index is empty -> needs training")
            return True
        else:
            logger.info(f"Index contains {doc_count} documents -> skipping training")
            return False
    except Exception as e:
        logger.warning(f"Could not check index status: {e}")
        return True


def main(args):
    """
    Main test function for MySQL + Elasticsearch SQL retrieval.
    
    Args:
        args: Command line arguments
    """
    logger.info("=" * 80)
    logger.info("SQL Retriever Test: MySQL + Elasticsearch")
    logger.info("=" * 80)
    
    # 1. Check for NVIDIA_API_KEY
    api_key = os.getenv("NVIDIA_API_KEY")
    if not api_key:
        logger.error("✗ NVIDIA_API_KEY environment variable is not set.")
        logger.error("  Please set it before running this script.")
        logger.error("  Example: export NVIDIA_API_KEY='nvapi-your-key-here'")
        sys.exit(1)
    logger.info("✓ NVIDIA_API_KEY is set")
    
    # 2. Check Elasticsearch connection
    if not check_elasticsearch_connection(args.es_url):
        logger.error("Failed to connect to Elasticsearch. Please ensure it's running.")
        logger.error(f"  Start Elasticsearch: docker run -p 9200:9200 -e 'discovery.type=single-node' docker.elastic.co/elasticsearch/elasticsearch:8.11.0")
        sys.exit(1)
    
    # 3. Check MySQL connection
    if not check_mysql_connection(
        args.mysql_host,
        args.mysql_user,
        args.mysql_password,
        args.mysql_database,
        args.mysql_port
    ):
        logger.error("Failed to connect to MySQL. Please check your connection parameters.")
        sys.exit(1)
    
    logger.info("-" * 80)
    logger.info("Configuration:")
    logger.info(f"  LLM Model: {args.llm_model}")
    logger.info(f"  Embedding Model: {args.embedding_model}")
    logger.info(f"  MySQL: {args.mysql_host}:{args.mysql_port}/{args.mysql_database}")
    logger.info(f"  Elasticsearch: {args.es_url}")
    logger.info(f"  ES Index: {args.es_index}")
    logger.info(f"  Training Data: {args.training_data}")
    logger.info("-" * 80)
    
    # 4. Create embedding function
    logger.info("Initializing NVIDIA embedding function...")
    embedding_function = NVIDIAEmbeddingFunction(
        api_key=api_key,
        model=args.embedding_model
    )
    
    # 5. Create ElasticNIMVanna instance
    logger.info("Initializing ElasticNIMVanna...")
    vn_instance = ElasticNIMVanna(
        VectorConfig={
            "url": args.es_url,
            "index_name": args.es_index,
            "username": args.es_username,
            "password": args.es_password,
            "api_key": args.es_api_key,
            "embedding_function": embedding_function,
        },
        LLMConfig={
            "api_key": api_key,
            "model": args.llm_model,
        }
    )
    
    # 6. Connect to MySQL database
    logger.info("Connecting to MySQL database...")
    mysql_connection_string = (
        f"mysql+pymysql://{args.mysql_user}:{args.mysql_password}"
        f"@{args.mysql_host}:{args.mysql_port}/{args.mysql_database}"
    )
    
    from sqlalchemy import create_engine
    engine = create_engine(mysql_connection_string)
    vn_instance.connect_to_sqlalchemy(engine)
    logger.info("✓ Connected to MySQL database")
    
    # 7. Check if training is needed
    from elasticsearch import Elasticsearch
    es_check_client = Elasticsearch(args.es_url)
    
    if needs_training(es_check_client, args.es_index):
        if args.training_data and os.path.exists(args.training_data):
            logger.info("Training vector store with provided data...")
            init_vanna(vn_instance, args.training_data)
            logger.info("✓ Training complete")
        else:
            logger.warning("⚠ No training data provided or file not found")
            logger.warning(f"  Looked for: {args.training_data}")
            logger.warning("  The model will work but may produce less accurate SQL")
    
    # 8. Test SQL generation
    logger.info("=" * 80)
    logger.info(f"Testing SQL Generation")
    logger.info(f"Question: {args.question}")
    logger.info("-" * 80)
    
    try:
        # Generate SQL
        logger.info("Generating SQL query...")
        sql_query = vn_instance.generate_sql(
            question=args.question,
            allow_llm_to_see_data=True
        )
        
        if not sql_query or sql_query.strip() == "":
            logger.error("✗ SQL generation returned an empty query")
            return
        
        logger.info("✓ SQL generated successfully:")
        logger.info("-" * 80)
        logger.info(sql_query)
        logger.info("-" * 80)
        
        # Execute SQL
        logger.info("Executing SQL query...")
        df = vn_instance.run_sql(sql_query)
        
        if df is not None and not df.empty:
            logger.info(f"✓ Query executed successfully. Retrieved {len(df)} rows")
            logger.info("=" * 80)
            logger.info("Results:")
            logger.info("=" * 80)
            print(df.to_string())
            logger.info("=" * 80)
            
            # Show summary statistics if numeric columns
            numeric_cols = df.select_dtypes(include=['number']).columns
            if len(numeric_cols) > 0:
                logger.info("\nSummary Statistics:")
                print(df[numeric_cols].describe())
                logger.info("=" * 80)
        elif df is not None and df.empty:
            logger.info("✓ Query executed successfully, but no results were returned")
        else:
            logger.warning("⚠ SQL execution did not return a DataFrame")
    
    except Exception as e:
        logger.error(f"✗ An error occurred: {e}")
        import traceback
        logger.error(traceback.format_exc())
        sys.exit(1)
    
    logger.info("\n✓ Test completed successfully!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Test SQL Retriever with MySQL and Elasticsearch",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Basic usage with defaults
    python test_mysql_elastic.py
    
    # Custom question
    python test_mysql_elastic.py -q "Show me the top 10 customers"
    
    # Full configuration
    python test_mysql_elastic.py \\
        --mysql-host localhost \\
        --mysql-user root \\
        --mysql-password mypassword \\
        --mysql-database mydb \\
        --es-url http://localhost:9200 \\
        --es-index my_sql_vectors \\
        -q "What are the sales by region?"
        """
    )
    
    # Question
    parser.add_argument(
        "-q", "--question",
        type=str,
        default="What are the top 10 best-selling products by quantity?",
        help="Natural language question to test"
    )
    
    # MySQL connection
    mysql_group = parser.add_argument_group("MySQL Configuration")
    mysql_group.add_argument(
        "--mysql-host",
        type=str,
        default="localhost",
        help="MySQL host (default: localhost)"
    )
    mysql_group.add_argument(
        "--mysql-port",
        type=int,
        default=3306,
        help="MySQL port (default: 3306)"
    )
    mysql_group.add_argument(
        "--mysql-user",
        type=str,
        default="root",
        help="MySQL username (default: root)"
    )
    mysql_group.add_argument(
        "--mysql-password",
        type=str,
        default="",
        help="MySQL password (default: empty)"
    )
    mysql_group.add_argument(
        "--mysql-database",
        type=str,
        default="test",
        help="MySQL database name (default: test)"
    )
    
    # Elasticsearch connection
    es_group = parser.add_argument_group("Elasticsearch Configuration")
    es_group.add_argument(
        "--es-url",
        type=str,
        default="http://localhost:9200",
        help="Elasticsearch URL (default: http://localhost:9200)"
    )
    es_group.add_argument(
        "--es-index",
        type=str,
        default="vanna_sql_vectors",
        help="Elasticsearch index name (default: vanna_sql_vectors)"
    )
    es_group.add_argument(
        "--es-username",
        type=str,
        default=None,
        help="Elasticsearch username (optional)"
    )
    es_group.add_argument(
        "--es-password",
        type=str,
        default=None,
        help="Elasticsearch password (optional)"
    )
    es_group.add_argument(
        "--es-api-key",
        type=str,
        default=None,
        help="Elasticsearch API key (optional)"
    )
    
    # Model configuration
    model_group = parser.add_argument_group("Model Configuration")
    model_group.add_argument(
        "--llm-model",
        type=str,
        default="meta/llama-3.1-70b-instruct",
        help="LLM model name (default: meta/llama-3.1-70b-instruct)"
    )
    model_group.add_argument(
        "--embedding-model",
        type=str,
        default="nvidia/llama-3.2-nv-embedqa-1b-v2",
        help="Embedding model name (default: nvidia/llama-3.2-nv-embedqa-1b-v2)"
    )
    
    # Training data
    parser.add_argument(
        "--training-data",
        type=str,
        default=None,
        help="Path to YAML training data file (optional)"
    )
    
    args = parser.parse_args()
    main(args)

