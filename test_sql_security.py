from app import validate_sql

tests = [
    'SELECT * FROM "Artist"',
    'DROP TABLE "Artist"',
    'SELECT 1; DROP TABLE "Artist"',
    'WITH artists AS (SELECT * FROM "Artist") SELECT * FROM artists'
]

for query in tests:
    print(f"\nQuery: {query}")
    print(f"Result: {validate_sql(query)}")