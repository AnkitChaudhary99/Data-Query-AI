import os
from pathlib import Path

import pandas as pd
from sqlalchemy import create_engine, inspect
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent
ROOT_DIR = BASE_DIR.parent

load_dotenv(ROOT_DIR / ".env")

DB_URL = os.getenv("DB_URL")

if not DB_URL:
    raise ValueError("DB_URL not found in .env file")

engine = create_engine(DB_URL)

file_list = [
    "Album.csv",
    "Artist.csv",
    "Customer.csv",
    "Employee.csv",
    "Genre.csv",
    "Invoice.csv",
    "InvoiceLine.csv",
    "MediaType.csv",
    "Playlist.csv",
    "PlaylistTrack.csv",
    "Track.csv"
]

try:
    with engine.connect() as conn:
        print("Successfully connected to PostgreSQL!")

    for file in file_list:
        file_path = BASE_DIR / file

        if not file_path.exists():
            print(f"File not found: {file}")
            continue

        df = pd.read_csv(file_path)
        table_name = file_path.stem

        df.to_sql(
            table_name,
            engine,
            if_exists="replace",
            index=False
        )

        print(f"Uploaded {table_name}: {len(df)} rows")

    inspector = inspect(engine)
    tables = inspector.get_table_names()

    print("\nDatabase tables:")
    for table in tables:
        print(f"- {table}")

    print(f"\nTotal tables created: {len(tables)}")

except Exception as e:
    print(f"Database deployment failed: {e}")

finally:
    engine.dispose()