import extract

# Create a single connection
conn = extract.connect_to_postgres(dbname="database",
                                   user="user",
                                   password="password")

# Use the same connection for multiple queries
if conn:
    query = extract.fetch_data_from_db("""SELECT p.*, a.hadm_id, a.admittime, a.dischtime, a.deathtime, 
                                                a.admission_type, a.diagnosis, a.hospital_expire_flag,
                                                d.seq_num, icd.icd9_code, icd.short_title, icd.long_title
                                        FROM mimiciii.patients p
                                        INNER JOIN mimiciii.admissions a ON p.subject_id = a.subject_id
                                        INNER JOIN mimiciii.diagnoses_icd d
                                        ON a.subject_id = d.subject_id
                                        AND a.hadm_id = d.hadm_id
                                        INNER JOIN mimiciii.d_icd_diagnoses icd
                                        ON icd.ICD9_CODE = d.ICD9_CODE
                                        """,
                                       conn)
    print(query.info())

    # Close connection after all queries
    conn.close()
