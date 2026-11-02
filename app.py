import streamlit as st
import pandas as pd
from sqlalchemy import create_engine, text
from langchain_google_genai import GoogleGenerativeAI
from dotenv import load_dotenv
from pathlib import Path
import re
import os
import json
import sqlglot
from sqlglot import exp


# ------------------- CONFIGURATION -------------------

BASE_DIR = Path(__file__).resolve().parent

load_dotenv(BASE_DIR / ".env")

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
DB_URL = os.getenv("DB_URL")

st.set_page_config(
    page_title="Data Query AI",
    page_icon="📊",
    layout="wide"
)

st.title("Data Query AI 📊")
st.caption("Ask questions about your PostgreSQL database using natural language.")


if not GOOGLE_API_KEY:
    st.error("GOOGLE_API_KEY is missing from your .env file.")
    st.stop()

if not DB_URL:
    st.error("DB_URL is missing from your .env file.")
    st.stop()


# ------------------- SESSION STATE -------------------

if "messages" not in st.session_state:
    st.session_state.messages = []


# ------------------- DATABASE -------------------

@st.cache_resource
def get_db_engine():
    return create_engine(
        DB_URL,
        pool_pre_ping=True
    )


def get_schema():

    engine = get_db_engine()

    inspector_query = text("""
        SELECT table_name, column_name
        FROM information_schema.columns
        WHERE table_schema = 'public'
        ORDER BY table_name, ordinal_position;
    """)

    schema_string = ""

    try:

        with engine.connect() as conn:

            result = conn.execute(inspector_query)

            current_table = None

            for row in result:

                table_name, column_name = row

                if table_name != current_table:

                    if current_table is not None:
                        schema_string += "\n"

                    schema_string += f'Table: "{table_name}"\n'

                    current_table = table_name

                schema_string += f'  - "{column_name}"\n'

    except Exception as e:

        st.error(f"Error fetching database schema: {e}")

        return ""

    return schema_string


# ------------------- LLM INITIALIZATION -------------------

@st.cache_resource
def get_llm():

    return GoogleGenerativeAI(
        model="gemini-3.5-flash",
        google_api_key=GOOGLE_API_KEY
    )


llm = get_llm()


# ------------------- SQL HELPERS -------------------

def clean_sql(sql_text: str):

    sql_text = re.sub(
        r"```sql\s*",
        "",
        sql_text,
        flags=re.IGNORECASE
    )

    sql_text = re.sub(
        r"```",
        "",
        sql_text
    )

    return sql_text.strip()


def validate_sql(sql_query: str):

    if not sql_query:
        return False, "SQL query is empty."

    query = clean_sql(sql_query).strip()

    try:

        parsed_queries = sqlglot.parse(
            query,
            read="postgres"
        )

        parsed_queries = [
            parsed
            for parsed in parsed_queries
            if parsed is not None
        ]

        if len(parsed_queries) != 1:

            return False, "Only one SQL statement is allowed."

        parsed = parsed_queries[0]

        # Only SELECT queries are allowed.
        # WITH queries are represented as SELECT expressions
        # containing CTEs in SQLGlot.

        if not isinstance(parsed, exp.Select):

            return False, "Only SELECT queries are allowed."

        # Reject SELECT INTO.
        if parsed.args.get("into") is not None:

            return False, "SELECT INTO is not allowed."

        # Reject SELECT statements that lock rows.
        if parsed.args.get("locks"):

            return False, "SELECT FOR UPDATE/SHARE is not allowed."

        # Reject data-modifying operations inside CTEs.
        for node in parsed.walk():

            if isinstance(
                node,
                (
                    exp.Insert,
                    exp.Update,
                    exp.Delete,
                    exp.Merge,
                    exp.Create,
                    exp.Drop,
                    exp.Alter,
                    exp.TruncateTable,
                )
            ):

                return False, "Data-modifying SQL operations are not allowed."

        # Reject SELECT statements with INTO or locking clauses
        # anywhere inside the parsed query tree.
        for node in parsed.walk():

            if isinstance(node, exp.Select):

                if node.args.get("into") is not None:

                    return False, "SELECT INTO is not allowed."

                if node.args.get("locks"):

                    return False, "SELECT locking clauses are not allowed."

        return True, "Valid read-only SELECT query."

    except sqlglot.errors.ParseError:

        return False, "SQL syntax is invalid."


def get_few_shot_examples(file_path="fewshots.json"):

    try:

        full_path = BASE_DIR / file_path

        if not full_path.exists():
            return ""

        with open(full_path, "r", encoding="utf-8") as f:

            examples = json.load(f)

        formatted_examples = (
            "\n### EXAMPLES OF QUESTIONS AND SQL QUERIES:\n"
        )

        for ex in examples:

            question = ex.get("naturalQuestion")
            sql = ex.get("sqlQuery")

            if question and sql:

                formatted_examples += (
                    f"Question: {question}\n"
                    f"SQL: {sql}\n\n"
                )

        return formatted_examples

    except Exception as e:

        st.error(f"Error loading few-shot examples: {e}")

        return ""


# ------------------- SQL GENERATION -------------------

def generate_sql_query(user_question, schema, chat_history):

    few_shots = get_few_shot_examples()

    history_text = ""

    for message in chat_history[-10:]:

        role = message["role"].upper()
        content = message["content"]

        history_text += f"{role}: {content}\n"

        if message.get("sql"):

            history_text += (
                f"SQL GENERATED: {message['sql']}\n"
            )

    prompt = f"""
You are an expert PostgreSQL Data Analyst.

Your task is to generate SQL queries based on the user's question.

DATABASE SCHEMA:
{schema}

CONVERSATION HISTORY:
{history_text if history_text else "No previous conversation."}

CURRENT USER QUESTION:
{user_question}

CONVERSATION CONTEXT RULES:

1. Carefully examine the previous conversation before generating SQL.
2. Determine whether the current question is a follow-up or a new question.
3. For follow-up questions, preserve the relevant filters, entities,
   conditions, and context from previous queries.
4. When the user asks to modify a previous query, modify that query
   instead of generating an unrelated query.
5. Resolve references such as "they", "them", "those", "their",
   "the first one", "the second one", and "what about the others".
6. If the user asks for additional information about previous results,
   use the appropriate tables and joins to retrieve that information.
7. If the user asks a completely unrelated question, generate SQL
   based on that new question instead of unnecessarily reusing context.
8. Never assume previous query results unless they are present
   in the conversation history.

SQL GENERATION RULES:

1. The database uses PostgreSQL.
2. ALWAYS surround ALL table names and column names with double quotes.
3. Use table aliases when joining tables with identical column names.
4. Generate only read-only SELECT queries.
5. Never generate INSERT, UPDATE, DELETE, DROP, ALTER, CREATE,
   TRUNCATE, or other data-modifying statements.
6. Return ONLY the SQL query without Markdown formatting or comments.
7. Use PostgreSQL syntax.
8. Do not invent table names or column names.
9. Never generate multiple SQL statements.
10. Prefer queries that return a reasonable number of rows.

{few_shots}

CURRENT QUESTION:
{user_question}

SQL:
"""

    try:

        response = llm.invoke(prompt)

        return clean_sql(response)

    except Exception as e:

        st.error(f"Error generating SQL query: {e}")

        return ""



# ------------------- NATURAL LANGUAGE RESPONSE -------------------

def get_natural_language_response(question, data_df):

    try:

        limited_data = data_df.head(15).to_markdown(index=False)

    except Exception:

        limited_data = data_df.head(15).to_string(index=False)

    prompt = f"""
You are a professional Data Analyst.

USER QUESTION:
{question}

SQL QUERY RESULTS:
{limited_data}

INSTRUCTIONS:

1. Answer the user's question directly and clearly.
2. Base your answer only on the provided data.
3. Include relevant numerical values.
4. If the results are empty, explain that no matching records were found.
5. Keep your response professional and concise.
"""

    try:

        response = llm.invoke(prompt)

        return response

    except Exception as e:

        st.error(f"Error generating natural language response: {e}")

        return "Error generating response."


# ------------------- QUERY EXECUTION -------------------

def execute_sql_query(sql_query):

    is_valid, validation_message = validate_sql(sql_query)

    if not is_valid:

        st.warning(
            f"Query rejected: {validation_message}"
        )

        return None

    try:

        engine = get_db_engine()

        with engine.connect() as conn:

            with conn.begin():

                # PostgreSQL read-only transaction.
                conn.exec_driver_sql(
                    "SET TRANSACTION READ ONLY"
                )

                # Prevent long-running queries.
                conn.exec_driver_sql(
                    "SET LOCAL statement_timeout = '15s'"
                )

                # Limit the number of returned rows.
                sql_query = clean_sql(sql_query).rstrip(";").strip()

                safe_query = f"""
                    SELECT *
                    FROM ({sql_query}) AS safe_result
                    LIMIT 1000
                """

                result_df = pd.read_sql(
                    text(safe_query),
                    conn
                )

        return result_df

    except Exception as e:

        st.error(f"Error executing SQL query: {e}")

        return None


# ------------------- STREAMLIT CHAT APPLICATION -------------------

if __name__ == "__main__":

    with st.spinner("Connecting to database..."):

        schema = get_schema()

    if not schema:

        st.error("Unable to retrieve database schema.")

        st.stop()

    st.success("Database connected successfully!")

    # Display previous conversation.

    for message in st.session_state.messages:

        with st.chat_message(message["role"]):

            st.markdown(message["content"])

            if message.get("sql"):

                with st.expander("View Generated SQL"):

                    st.code(
                        message["sql"],
                        language="sql"
                    )

            if message.get("data") is not None:

                st.dataframe(
                    message["data"],
                    use_container_width=True
                )

    # Chat input.

    user_question = st.chat_input(
        "Ask a question about your database..."
    )

    if user_question:

        with st.chat_message("user"):

            st.markdown(user_question)

        st.session_state.messages.append({
            "role": "user",
            "content": user_question
        })

        with st.chat_message("assistant"):

            with st.spinner("Generating SQL query..."):

                sql_query = generate_sql_query(
                    user_question,
                    schema,
                    st.session_state.messages[:-1]
                )

            if sql_query:

                st.markdown("**Generated SQL Query:**")

                st.code(
                    sql_query,
                    language="sql"
                )

                is_valid, validation_message = validate_sql(sql_query)

                if not is_valid:

                    st.error(
                        f"SQL validation failed: {validation_message}"
                    )

                    answer = (
                        "The generated SQL query failed security validation."
                    )

                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer,
                        "sql": sql_query
                    })

                else:

                    with st.spinner("Executing SQL query..."):

                        result_df = execute_sql_query(sql_query)

                    if result_df is not None:

                        st.markdown("**Query Results:**")

                        st.dataframe(
                            result_df,
                            use_container_width=True
                        )

                        with st.spinner("Analyzing results..."):

                            answer = get_natural_language_response(
                                user_question,
                                result_df
                            )

                        st.markdown("**Answer:**")

                        st.markdown(answer)

                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": answer,
                            "sql": sql_query,
                            "data": result_df
                        })

                    else:

                        answer = "SQL execution failed."

                        st.error(answer)

                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": answer,
                            "sql": sql_query
                        })

            else:

                answer = "Unable to generate SQL query."

                st.error(answer)

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer
                })