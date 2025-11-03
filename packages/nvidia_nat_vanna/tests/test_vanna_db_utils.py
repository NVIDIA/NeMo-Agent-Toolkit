# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# pylint: disable=unused-argument

from unittest.mock import MagicMock
from unittest.mock import patch

import pytest
from pydantic import BaseModel

from nat.plugins.vanna.db_utils import QueryResult
from nat.plugins.vanna.db_utils import SupportedDatabase
from nat.plugins.vanna.db_utils import connect_to_database
from nat.plugins.vanna.db_utils import connect_to_databricks
from nat.plugins.vanna.db_utils import execute_query
from nat.plugins.vanna.db_utils import extract_sql_from_message
from nat.plugins.vanna.db_utils import setup_vanna_db_connection


class TestQueryResult:
    """Test QueryResult model."""

    def test_to_records(self):
        """Test conversion to list of dictionaries."""
        result = QueryResult(results=[(1, "test"), (2, "data")], column_names=["id", "name"])

        records = result.to_records()
        assert records == [{"id": 1, "name": "test"}, {"id": 2, "name": "data"}]

    def test_to_dataframe(self):
        """Test conversion to pandas DataFrame."""
        result = QueryResult(results=[(1, "test"), (2, "data")], column_names=["id", "name"])

        df = result.to_dataframe()
        assert len(df) == 2
        assert list(df.columns) == ["id", "name"]
        assert df.iloc[0]["id"] == 1
        assert df.iloc[0]["name"] == "test"

    def test_empty_result(self):
        """Test empty QueryResult."""
        result = QueryResult(results=[], column_names=[])
        assert result.row_count == 0
        assert result.to_records() == []


class TestExtractSqlFromMessage:
    """Test SQL extraction from various formats."""

    def test_basemodel_with_sql_field(self):
        """Test BaseModel with sql field."""

        class MockSQLOutput(BaseModel):
            sql: str
            explanation: str | None = None

        model = MockSQLOutput(sql="SELECT * FROM users", explanation="Get all users")
        assert extract_sql_from_message(model) == "SELECT * FROM users"

    def test_dict_with_sql_key(self):
        """Test dictionary with sql key."""
        data = {"sql": "SELECT * FROM users", "explanation": "Get all users"}
        assert extract_sql_from_message(data) == "SELECT * FROM users"

    def test_json_string(self):
        """Test JSON string with sql key."""
        json_str = '{"sql": "SELECT * FROM users", "explanation": "Get all users"}'
        assert extract_sql_from_message(json_str) == "SELECT * FROM users"

    def test_sql_equals_format(self):
        """Test sql='...' format."""
        text = "sql='SELECT * FROM users' explanation='Get all users'"
        assert extract_sql_from_message(text) == "SELECT * FROM users"

    def test_sql_equals_double_quotes(self):
        """Test sql=\"...\" format."""
        text = 'sql="SELECT * FROM users" explanation="Get all users"'
        assert extract_sql_from_message(text) == "SELECT * FROM users"

    def test_tool_message_format(self):
        """Test extraction from tool message format."""
        message = 'content="SELECT * FROM users"'
        assert extract_sql_from_message(message) == "SELECT * FROM users"

    def test_object_with_content_attribute(self):
        """Test object with content attribute."""

        class MockMessage:

            def __init__(self, content):
                self.content = content

        msg = MockMessage(content={"sql": "SELECT * FROM users"})
        assert extract_sql_from_message(msg) == "SELECT * FROM users"


class TestConnectToDatabricks:
    """Test Databricks connection."""

    @patch("sqlalchemy.create_engine")
    def test_connection_error_propagation(self, mock_create_engine):
        """Test connection errors are properly propagated."""
        mock_create_engine.side_effect = ValueError("Invalid connection string")

        with pytest.raises(ValueError, match="Invalid connection string"):
            connect_to_databricks("invalid://url")


class TestConnectToDatabase:
    """Test database connection."""

    @patch("nat.plugins.vanna.db_utils.connect_to_databricks")
    @pytest.mark.parametrize(
        "db_type",
        ["databricks", "DATABRICKS", SupportedDatabase.DATABRICKS],
        ids=["lowercase_string", "uppercase_string", "enum"],
    )
    def test_databricks_connection(self, mock_databricks, db_type):
        """Test connection with various databricks type formats."""
        mock_connection = MagicMock()
        mock_databricks.return_value = mock_connection

        result = connect_to_database(db_type, "databricks://token@host/db")
        assert result == mock_connection
        mock_databricks.assert_called_once_with(connection_url="databricks://token@host/db")

    @pytest.mark.parametrize(
        "invalid_type,expected_msg",
        [
            ("mysql", "Unsupported database type: 'mysql'"),
            ("postgres", "Unsupported database type: 'postgres'"),
            ("", "Unsupported database type: ''"),
        ],
    )
    def test_unsupported_database_types(self, invalid_type, expected_msg):
        """Test error messages for various unsupported database types."""
        with pytest.raises(ValueError, match=expected_msg):
            connect_to_database(invalid_type, "connection_url")


class TestExecuteQuery:
    """Test query execution."""

    def test_successful_query_with_results(self):
        """Test query execution returns correct QueryResult with data."""
        mock_connection = MagicMock()
        mock_conn = MagicMock()
        mock_result = MagicMock()

        # Setup mock chain
        mock_connection.connect.return_value.__enter__.return_value = mock_conn
        mock_conn.execute.return_value = mock_result
        mock_result.fetchall.return_value = [(1, "alice", 25), (2, "bob", 30)]
        mock_result.keys.return_value = ["id", "name", "age"]

        result = execute_query(mock_connection, "SELECT id, name, age FROM users")

        assert isinstance(result, QueryResult)
        assert result.row_count == 2
        assert result.column_names == ["id", "name", "age"]
        assert result.results[0] == (1, "alice", 25)

    def test_empty_query_result(self):
        """Test query that returns no rows."""
        mock_connection = MagicMock()
        mock_conn = MagicMock()
        mock_result = MagicMock()

        mock_connection.connect.return_value.__enter__.return_value = mock_conn
        mock_conn.execute.return_value = mock_result
        mock_result.fetchall.return_value = []
        mock_result.keys.return_value = ["id", "name"]

        result = execute_query(mock_connection, "SELECT * FROM users WHERE id = 999")

        assert result.row_count == 0
        assert result.column_names == ["id", "name"]
        assert result.to_records() == []

    def test_query_execution_error(self):
        """Test database errors are properly propagated."""
        mock_connection = MagicMock()
        mock_connection.connect.side_effect = RuntimeError("Connection lost")

        with pytest.raises(RuntimeError, match="Connection lost"):
            execute_query(mock_connection, "SELECT * FROM users")


class TestSetupVannaDbConnection:
    """Test Vanna database setup."""

    @patch("nat.plugins.vanna.db_utils.connect_to_database")
    def test_vanna_configuration(self, mock_connect):
        """Test Vanna instance is properly configured with database connection."""
        mock_connection = MagicMock()
        mock_connect.return_value = mock_connection
        mock_vanna = MagicMock()
        # Ensure db_engine doesn't exist initially so connect_to_database gets called
        del mock_vanna.db_engine

        setup_vanna_db_connection(mock_vanna, SupportedDatabase.DATABRICKS, "databricks://token@host/db")

        # Verify vanna is configured with the connection
        assert hasattr(mock_vanna, "db_engine")
        assert hasattr(mock_vanna, "run_sql")
        assert mock_vanna.run_sql_is_set is True
        mock_connect.assert_called_once_with(database_type=SupportedDatabase.DATABRICKS,
                                             connection_url="databricks://token@host/db")

    @pytest.mark.asyncio
    @patch("nat.plugins.vanna.db_utils.async_execute_query")
    @patch("nat.plugins.vanna.db_utils.connect_to_database")
    async def test_vanna_run_sql_integration(self, mock_connect, mock_async_execute):
        """Test the dynamically created run_sql function executes queries and returns DataFrames."""

        mock_connection = MagicMock()
        mock_connect.return_value = mock_connection

        # Mock async_execute_query to return a QueryResult
        mock_query_result = QueryResult(results=[(100, "product_a"), (200, "product_b")],
                                        column_names=["price", "name"])
        mock_async_execute.return_value = mock_query_result

        mock_vanna = MagicMock()
        setup_vanna_db_connection(mock_vanna, "databricks", "databricks://token@host/db")

        # Get the actual run_sql function that was assigned to the mock
        run_sql_func = mock_vanna.run_sql

        # Execute query through the actual run_sql function
        df = await run_sql_func("SELECT price, name FROM products")

        # Verify DataFrame structure
        assert len(df) == 2
        assert list(df.columns) == ["price", "name"]
        assert df.iloc[0]["price"] == 100
        assert df.iloc[1]["name"] == "product_b"
