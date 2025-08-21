import psycopg2
from psycopg2.extras import RealDictCursor
import streamlit as st

@st.cache_resource
def get_db_params():
    return {
        "host": st.secrets["postgres"]["host"],
        "dbname": st.secrets["postgres"]["dbname"],
        "user": st.secrets["postgres"]["user"],
        "password": st.secrets["postgres"]["password"],
        "port": st.secrets["postgres"]["port"],
        "cursor_factory": RealDictCursor
    }

def get_connection():
    params = get_db_params()
    return psycopg2.connect(**params)
