"""Global test configuration.

Ensures all tests use a temporary SQLite database instead of the production
file-based database.

The DB_URL env var MUST be set before db.engine is first imported, because
db/engine.py reads it at module level. init_db() is also called here at
module level so that tables exist when web/app.py's module-level
task_store = TaskStoreDB() tries to use the DB.

NOTE: We use a temp FILE (not :memory:) because in-memory SQLite creates
separate DBs per connection — tables created by init_db() won't be visible
to sessions created later by TaskStoreDB.
"""

import os
import tempfile

# Create a temp file for the test database
_TEST_DB_FD, _TEST_DB_PATH = tempfile.mkstemp(suffix=".db")
os.close(_TEST_DB_FD)

# Set DB_URL BEFORE any db.engine or web.app imports
os.environ["DB_URL"] = f"sqlite:///{_TEST_DB_PATH}"

from db.engine import init_db, reset_engine  # noqa: E402

# Initialize tables at module level so they exist when web.app is imported
reset_engine()
init_db(url=f"sqlite:///{_TEST_DB_PATH}")
