import pandas as pd
import psycopg2


def connect_to_postgres(dbname, user, password, host='localhost', port='5432'):
    """
    Connects to a PostgresSQL database and returns the connection object.

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


def fetch_data_from_db(query, db_name, user, password, host, port):
    """
    Connects to a PostgresSQL database, executes the given SQL query, and returns the result as a Pandas DataFrame.

    Parameters:
    - query (str): The SQL query to execute.
    - db_name (str): Name of the PostgresSQL database.
    - user (str): Database username.
    - password (str): Database password.
    - host (str): Database host (e.g., 'localhost').
    - port (str/int): Database port (e.g., 5432).

    Returns:
    - pd.DataFrame: The query results as a Pandas DataFrame.
    """

    cursor = None  # Initialize cursor to None to avoid reference before assignment error

    try:
        # Establish database connection
        conn = connect_to_postgres(db_name, user, password, host, port)

    except psycopg2.OperationalError as op_err:
        print(f"Operational Error: {op_err} (Check database connection settings!)")
        return None  # Stop function execution early if connection fails

    try:
        # Create cursor after successful connection
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
        # Close the cursor and connection if they were successfully created
        if cursor:
            cursor.close()
        if 'conn' in locals() and conn:
            conn.close()

    return df


