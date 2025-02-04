import pandas as pd
import psycopg2


def connect_to_postgres(dbname, user, password, host='localhost', port='5432'):
    """
    Connects to a PostgresSQL database and returns the connection object
    :param dbname: Name of the database
    :param user: Username for authentication
    :param password: Password for authentication
    :param host: Host where the database is running (default: localhost)
    :param port: Port on which the database is listening (default: 5432)
    :return: Connection object if successful, None otherwise
    """
    try:
        # Attempt to establish a connection to PostgreSQL
        conn = psycopg2.connect(
            dbname=dbname,
            user=user,
            password=password,
            host=host,
            port=port
        )
        print("Connected to the database successfully!")
        return conn

    except psycopg2.OperationalError as op_err:
        print(f"Operational Error: {op_err} (Check database connection details or server availability.)")
        return None

    except psycopg2.InterfaceError as int_err:
        print(f"Interface Error: {int_err} (There was an issue with the database interface.)")
        return None

    except psycopg2.DatabaseError as db_err:
        print(f"Database Error: {db_err} (There was an issue with the database.)")
        return None

    except Exception as e:
        print(f"Unexpected Error: {e} (Something went wrong while connecting to the database.)")
        return None


def fetch_data_from_db(query, conn):
    """
    Executes a given SQL query using an existing PostgreSQL connection and returns the result as a Pandas DataFrame.

    Parameters:
    - query (str): The SQL query to execute.
    - conn (psycopg2.connection): Active database connection.

    Returns:
    - pd.DataFrame: Query results as a Pandas DataFrame.
    """

    cursor = None
    try:
        # Ensure connection is valid before proceeding
        if conn is None or conn.closed:
            print("Error: Database connection is not available.")
            return None

        # Create a cursor
        cursor = conn.cursor()

        # Execute SQL query
        cursor.execute(query)

        # Fetch all results
        results = cursor.fetchall()

        # Extract column names from cursor description
        columns = [col[0] for col in cursor.description]

        # Convert results to Pandas DataFrame
        df = pd.DataFrame(results, columns=columns)

    except psycopg2.ProgrammingError as prog_err:
        print(f"SQL Programming Error: {prog_err} (Check your SQL syntax!)")
        df = None

    except psycopg2.DatabaseError as db_err:
        print(f"Database Error: {db_err} (Something went wrong with the database!)")
        df = None

    except Exception as e:
        print(f"Unexpected Error: {e}")
        df = None

    finally:
        # Close the cursor but keep the connection open
        if cursor:
            cursor.close()

    return df



