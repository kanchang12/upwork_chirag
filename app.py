

from flask import Flask, request, render_template, jsonify
import os
import pandas as pd
import openai
import re

# Initialize Flask app
app = Flask(__name__)
openai.api_key = os.getenv('OPENAI_API_KEY')

app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024 

# Global variable to store the uploaded data
global_data = None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['GET', 'POST'])
def upload_file():
    global global_data

    if 'document' not in request.files:
        return jsonify({"error": "No file part in the request"}), 400

    file = request.files['document']

    if file.filename == '':
        return jsonify({"error": "No file selected"}), 400

    if file and file.filename.endswith('.csv'):
        try:
            filename = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filename)
            global_data = pd.read_csv(filename)
            
            return jsonify({
                "message": "File uploaded successfully",
                "columns": global_data.columns.tolist()
            }), 200
        except Exception as e:
            return jsonify({"error": f"Error processing file: {str(e)}"}), 500
    else:
        return jsonify({"error": "Invalid file format. Please upload a CSV file."}), 400

def is_pandas_query(resp):
    pattern = r'global_data\.[a-zA-Z_]+\([^\)]*\)|global_data\[[\'"][^\'"]+[\'"]\]'
    return bool(re.search(pattern, resp))

def generate_query_with_openai(user_input, column_names):
    column_list = column_names if isinstance(column_names, list) else column_names.split(", ")
    
    SYSTEM_PROMPT = f"""
Data: You're working with a pandas DataFrame named global_data.

Available Columns:
{', '.join(column_list)}
country is country
population is Patient Incidence
Roche Recruitment Rate is roche factor or any other name

Critical Rules:
PLEASE RETURN ALL THE FOUR COLUMNS WHEN NOTHING IS SPECIFIED.
QUERY_START & QUERY_END: Each query must be enclosed within QUERY_START and QUERY_END. The query itself should be executable (no comments within the query).
global_data.assign(): Use this function to create new columns.
Exact Column Names: ALWAYS use the exact column names provided in the list above. Enclose them in square brackets and double quotes (e.g., global_data["Column Name"]).
Conditional Sorting: Use if or where statements within your queries for sorting based on conditions.
Column Selection: End your queries by selecting relevant columns using double brackets [].
Numeric Conversions: Use pd.to_numeric(global_data["Column Name"], errors='coerce') to convert columns to numeric format before sorting or calculations.
Unique Values: Use .nunique() to count unique values within a column.
Multiple Queries: Separate multiple queries with a new line. Each query should still follow the QUERY_START and QUERY_END format.
Column Names: Verify column names against the provided list and use exact matches.
Explanations: If needed, provide an EXPLANATION: section after your queries to explain the results.
Population: Use "Patient Incidence" for population-related queries.
Sorting: Unless specified otherwise, sort numerical data in descending order.
Unexpected Results: Double-check data types if results are unexpected and provide explanations.

MAKE NOTE OF IT
YOU MUST CHECK THE COLUMN {', '.join(column_list)}
If NOTHING IS SPECIFIED, ALL COLUMNS WILL BE RETURNED.
THE QUERY SHOULD BE EXECUTABLE REMOVE DIFFERENT THINGS
ADD EXPLANATIONS AFTER THE DATA

Example: if nothing mentioned, all columns will be returned

QUERY_START
global_data[{', '.join([f'"{col}"' for col in column_list])}]
QUERY_END

Remember:
Accurate column names are crucial for successful queries.
You can submit multiple queries separated by new lines, each with QUERY_START and QUERY_END.
You are adding some extra values resulting in crashes. Please make sure only pandas query to be sent, nothing else.

If user says "hi, hello, greet properly", respond logically.
"""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_input}
    ]
    
    response = openai.chat.completions.create(
        model="gpt-3.5-turbo-16k",
        messages=messages,
        temperature=0,
        max_tokens=450
    )
    resp = response.choices[0].message.content
    print(f"DEBUG: Response: {resp}")
    
    # Verify and correct column names
    for col in column_list:
        resp = resp.replace(f'["{col.strip()}"]', f'["{col}"]')
    print(f"DEBUG: Corrected column names: {resp}")
    return resp

def process_query(query):
    try:
        result = eval(query, {"global_data": global_data, "pd": pd})
        return result
    except Exception as e:
        print(f"DEBUG: Error executing query: {query}")
        print(f"DEBUG: Exception: {str(e)}")
        return None

def extract_pandas_queries(text):
    queries = []
    current_query = ""
    in_query = False

    for line in text.splitlines():
        line = line.strip()

        if line.startswith("QUERY_START"):
            in_query = True
            current_query = ""
        elif line.endswith("QUERY_END"):
            in_query = False
            if current_query:
                queries.append(current_query.strip())
            current_query = ""
        elif in_query:
            current_query += line + "\n"

    return queries

def extract_sort_columns(query):
    """
    Extracts columns to sort by from the query string.
    Assumes that the sorting columns are specified in a 'sort_values' function call.
    """
    sort_columns = []
    ascending = False  # Default sorting order is descending
    match = re.search(r'\bsort_values\s*\(\s*by\s*=\s*\[([^\]]+)\]', query)
    if match:
        sort_columns = [col.strip().strip('"').strip("'") for col in match.group(1).split(',')]
        ascending_match = re.search(r'ascending\s*=\s*(True|False)', query)
        if ascending_match:
            ascending = ascending_match.group(1) == 'True'
    return sort_columns, ascending

def enforce_sorting(df, query):
    """
    Enforces sorting on the DataFrame based on the query.
    """
    sort_columns, ascending = extract_sort_columns(query)
    if sort_columns:
        return df.sort_values(by=sort_columns, ascending=ascending)
    return df

def verify_and_execute_queries(queries):
    global global_data
    results = []

    for index, query in enumerate(queries):
        try:
            print(f"DEBUG: Processing query {index + 1}/{len(queries)}: {query}")
            # Process the query
            processed_query = process_query(query)
            if processed_query is not None:
                print(f"DEBUG: Query processed successfully: {query}")
                if isinstance(processed_query, pd.DataFrame):
                    # Enforce sorting if requested
                    processed_query = enforce_sorting(processed_query, query)
                    result = processed_query.dropna(axis=1, how='any').to_dict(orient='records')
                elif isinstance(processed_query, pd.Series):
                    # Enforce sorting if requested
                    if "sort" in query.lower():
                        processed_query = processed_query.sort_values(ascending=False)
                    result = processed_query.to_dict()
                else:
                    result = str(processed_query)
                
                results.append(result)
            else:
                print(f"DEBUG: Processed query is None: {query}")
        except Exception as e:
            print(f"DEBUG: Query failed: {query}")
            print(f"DEBUG: Exception: {str(e)}")
            # Skip this query and continue with the next one

    return results

@app.route('/chat', methods=['GET', 'POST'])
def chat():
    global global_data

    if global_data is None:
        return jsonify({"error": "No data uploaded. Please upload a CSV file first."}), 400

    user_input = request.json.get('message')
    
    if not user_input:
        return jsonify({"error": "No message provided"}), 400

    ai_response = generate_query_with_openai(user_input, global_data.columns.tolist())
    
    pandas_queries = extract_pandas_queries(ai_response)
    
    if not pandas_queries:
        # If no pandas queries found, return the original AI response
        return jsonify({"results": [{"message": ai_response}]}), 200
    
    query_results = verify_and_execute_queries(pandas_queries)
    
    return jsonify({"results": query_results}), 200

if __name__ == '__main__':
    app.run(debug=True)
