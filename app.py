import streamlit as st
import pandas as pd
import plotly.express as px
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

if "chart_settings" not in st.session_state:
    st.session_state.chart_settings = {}


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

        if not isinstance(parsed, exp.Select):

            return False, "Only SELECT queries are allowed."

        if parsed.args.get("into") is not None:

            return False, "SELECT INTO is not allowed."

        if parsed.args.get("locks"):

            return False, "SELECT FOR UPDATE/SHARE is not allowed."

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

        for node in parsed.walk():

            if isinstance(node, exp.Select):

                if node.args.get("into") is not None:

                    return False, "SELECT INTO is not allowed."

                if node.args.get("locks"):

                    return False, "SELECT locking clauses are not allowed."

        return True, "Valid read-only SELECT query."

    except sqlglot.errors.ParseError:

        return False, "SQL syntax is invalid."


# ------------------- FEW-SHOT EXAMPLES -------------------

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
3. For follow-up questions, preserve relevant filters, entities,
   conditions, and context from previous queries.
4. When the user asks to modify a previous query, modify that query.
5. Resolve references such as "they", "them", "those", "their",
   "the first one", "the second one", and "what about the others".
6. If the user asks for additional information about previous results,
   use appropriate tables and joins.
7. If the question is unrelated, generate a new query.
8. Never assume previous query results unless present in history.

SQL GENERATION RULES:

1. Use PostgreSQL syntax.
2. ALWAYS surround ALL table names and column names with double quotes.
3. Use aliases when joining tables with identical column names.
4. Generate only read-only SELECT queries.
5. Never generate data-modifying statements.
6. Return ONLY SQL without Markdown formatting or comments.
7. Do not invent table names or column names.
8. Never generate multiple SQL statements.
9. Prefer queries that return a reasonable number of rows.

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


# ------------------- SQL ERROR CORRECTION -------------------

def correct_sql_query(user_question, schema, failed_sql, error_message):

    prompt = f"""
You are an expert PostgreSQL SQL debugger.

The SQL query generated for the user's question failed.

USER QUESTION:
{user_question}

DATABASE SCHEMA:
{schema}

FAILED SQL QUERY:
{failed_sql}

POSTGRESQL ERROR:
{error_message}

INSTRUCTIONS:

1. Analyze the PostgreSQL error carefully.
2. Identify the exact problem in the failed SQL.
3. Correct the SQL using the provided database schema.
4. Preserve the original intent of the user's question.
5. Return ONLY the corrected SQL query.
6. Generate only read-only SELECT queries.
7. Never generate multiple SQL statements.
8. Use PostgreSQL syntax.
9. Do not invent table names or column names.
10. Return no Markdown code fences or explanations.

CORRECTED SQL:
"""

    try:

        response = llm.invoke(prompt)

        return clean_sql(response)

    except Exception as e:

        st.error(f"Error correcting SQL query: {e}")

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
4. If results are empty, explain that no matching records were found.
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

        return None, validation_message

    try:

        engine = get_db_engine()

        with engine.connect() as conn:

            with conn.begin():

                conn.exec_driver_sql(
                    "SET TRANSACTION READ ONLY"
                )

                conn.exec_driver_sql(
                    "SET LOCAL statement_timeout = '15s'"
                )

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

        return result_df, None

    except Exception as e:

        return None, str(e)


# ------------------- VISUALIZATION HELPERS -------------------

def get_chart_recommendation(data_df):

    if data_df.empty or len(data_df.columns) < 2:
        return None

    numeric_columns = data_df.select_dtypes(
        include=["number"]
    ).columns.tolist()

    categorical_columns = data_df.select_dtypes(
        exclude=["number", "datetime"]
    ).columns.tolist()

    datetime_columns = data_df.select_dtypes(
        include=["datetime"]
    ).columns.tolist()

    if not numeric_columns:
        return None

    if datetime_columns:
        return {
            "chart_type": "Line",
            "x_column": datetime_columns[0],
            "y_columns": [numeric_columns[0]]
        }

    if categorical_columns:
        return {
            "chart_type": "Bar",
            "x_column": categorical_columns[0],
            "y_columns": [numeric_columns[0]]
        }

    if len(numeric_columns) >= 2:
        return {
            "chart_type": "Scatter",
            "x_column": numeric_columns[0],
            "y_columns": [numeric_columns[1]]
        }

    return None


def render_visualization(data_df, message_index):

    if data_df.empty:

        st.info("No data available for visualization.")

        return

    columns = data_df.columns.tolist()

    numeric_columns = data_df.select_dtypes(
        include=["number"]
    ).columns.tolist()

    if not numeric_columns:

        st.info(
            "This result does not contain numeric columns suitable for charting."
        )

        return

    recommendation = get_chart_recommendation(data_df)

    if recommendation is None:

        recommendation = {
            "chart_type": "Bar",
            "x_column": columns[0],
            "y_columns": [numeric_columns[0]]
        }

    settings_key = str(message_index)

    if settings_key not in st.session_state.chart_settings:

        st.session_state.chart_settings[settings_key] = recommendation.copy()

    settings = st.session_state.chart_settings[settings_key]

    chart_types = [
        "Bar",
        "Line",
        "Scatter",
        "Pie",
        "Area",
        "Histogram"
    ]

    st.markdown("#### Interactive Visualization")

    chart_type = st.selectbox(
        "Chart Type",
        chart_types,
        index=chart_types.index(settings["chart_type"])
        if settings["chart_type"] in chart_types
        else 0,
        key=f"chart_type_{settings_key}"
    )

    settings["chart_type"] = chart_type

    if chart_type == "Histogram":

        x_column = st.selectbox(
            "Column",
            columns,
            index=columns.index(settings["x_column"])
            if settings["x_column"] in columns
            else 0,
            key=f"hist_x_{settings_key}"
        )

        settings["x_column"] = x_column

        fig = px.histogram(
            data_df,
            x=x_column,
            title=f"Distribution of {x_column}"
        )

    else:

        x_options = columns

        x_column = st.selectbox(
            "X-axis",
            x_options,
            index=x_options.index(settings["x_column"])
            if settings["x_column"] in x_options
            else 0,
            key=f"x_axis_{settings_key}"
        )

        settings["x_column"] = x_column

        if chart_type == "Pie":

            y_column = st.selectbox(
                "Values",
                numeric_columns,
                index=numeric_columns.index(settings["y_columns"][0])
                if settings["y_columns"][0] in numeric_columns
                else 0,
                key=f"pie_y_{settings_key}"
            )

            settings["y_columns"] = [y_column]

            fig = px.pie(
                data_df,
                names=x_column,
                values=y_column,
                title=f"{y_column} by {x_column}"
            )

        else:

            y_columns = st.multiselect(
                "Y-axis columns",
                numeric_columns,
                default=[
                    col for col in settings["y_columns"]
                    if col in numeric_columns
                ] or [numeric_columns[0]],
                key=f"y_axis_{settings_key}"
            )

            if not y_columns:

                st.warning("Select at least one numeric Y-axis column.")

                return

            settings["y_columns"] = y_columns

            if chart_type == "Bar":

                fig = px.bar(
                    data_df,
                    x=x_column,
                    y=y_columns,
                    title=f"{', '.join(y_columns)} by {x_column}",
                    barmode="group"
                )

            elif chart_type == "Line":

                fig = px.line(
                    data_df,
                    x=x_column,
                    y=y_columns,
                    title=f"{', '.join(y_columns)} over {x_column}",
                    markers=True
                )

            elif chart_type == "Scatter":

                fig = px.scatter(
                    data_df,
                    x=x_column,
                    y=y_columns[0],
                    title=f"{y_columns[0]} vs {x_column}"
                )

            elif chart_type == "Area":

                fig = px.area(
                    data_df,
                    x=x_column,
                    y=y_columns,
                    title=f"{', '.join(y_columns)} over {x_column}"
                )

    fig.update_layout(
        template="plotly_white",
        height=500,
        margin=dict(l=30, r=30, t=60, b=30),
        legend_title_text="Metrics"
    )

    st.plotly_chart(
        fig,
        use_container_width=True,
        key=f"plot_{settings_key}"
    )


# ------------------- STREAMLIT CHAT APPLICATION -------------------

if __name__ == "__main__":

    with st.spinner("Connecting to database..."):

        schema = get_schema()

    if not schema:

        st.error("Unable to retrieve database schema.")

        st.stop()

    st.success("Database connected successfully!")

    # Display previous conversation.

    for message_index, message in enumerate(st.session_state.messages):

        with st.chat_message(message["role"]):

            st.markdown(message["content"])

            if message.get("sql"):

                with st.expander("View Generated SQL"):

                    st.code(
                        message["sql"],
                        language="sql"
                    )

            if message.get("data") is not None:

                result_df = message["data"]

                table_tab, chart_tab = st.tabs(
                    ["📋 Results Table", "📊 Visualization"]
                )

                with table_tab:

                    st.dataframe(
                        result_df,
                        use_container_width=True
                    )

                    csv_data = result_df.to_csv(
                        index=False
                    ).encode("utf-8")

                    st.download_button(
                        "Download CSV",
                        csv_data,
                        file_name="query_results.csv",
                        mime="text/csv",
                        key=f"download_old_{message_index}"
                    )

                with chart_tab:

                    render_visualization(
                        result_df,
                        message_index
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

                        result_df, execution_error = execute_sql_query(
                            sql_query
                        )

                    # Automatic SQL correction attempt.

                    if result_df is None and execution_error:

                        st.warning(
                            "SQL execution failed. Attempting automatic correction..."
                        )

                        with st.spinner("Gemini is correcting the SQL query..."):

                            corrected_sql = correct_sql_query(
                                user_question,
                                schema,
                                sql_query,
                                execution_error
                            )

                        if corrected_sql:

                            corrected_valid, corrected_message = validate_sql(
                                corrected_sql
                            )

                            if corrected_valid:

                                st.markdown("**Corrected SQL Query:**")

                                st.code(
                                    corrected_sql,
                                    language="sql"
                                )

                                with st.spinner("Retrying SQL execution..."):

                                    result_df, retry_error = execute_sql_query(
                                        corrected_sql
                                    )

                                if result_df is None:

                                    execution_error = retry_error

                                else:

                                    sql_query = corrected_sql

                            else:

                                execution_error = corrected_message

                    if result_df is not None:

                        st.markdown("**Query Results:**")

                        table_tab, chart_tab = st.tabs(
                            ["📋 Results Table", "📊 Visualization"]
                        )

                        with table_tab:

                            st.dataframe(
                                result_df,
                                use_container_width=True
                            )

                            csv_data = result_df.to_csv(
                                index=False
                            ).encode("utf-8")

                            st.download_button(
                                "Download CSV",
                                csv_data,
                                file_name="query_results.csv",
                                mime="text/csv",
                                key="download_current"
                            )

                        with chart_tab:

                            render_visualization(
                                result_df,
                                len(st.session_state.messages)
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

                        answer = "SQL execution failed after correction attempt."

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
