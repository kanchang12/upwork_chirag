

from flask import Flask, request, render_template, jsonify
import os
import pandas as pd
import openai
import re
import json

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
        print("DEBUG: No file part in the request")
        return jsonify({"error": "No file part in the request"}), 400

    file = request.files['document']

    if file.filename == '':
        print("DEBUG: No file selected")
        return jsonify({"error": "No file selected"}), 400

    if file and file.filename.endswith('.csv'):
        try:
            filename = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filename)
            global_data = pd.read_csv(filename)
            print("GLOBAL DATA: ", global_data)
            print(f"DEBUG: File uploaded successfully: {filename}")
            return jsonify({
                "message": "File uploaded successfully",
                "columns": global_data.columns.tolist()
            }), 200
        except Exception as e:
            print(f"DEBUG: Error processing file: {str(e)}")
            return jsonify({"error": f"Error processing file: {str(e)}"}), 500
    else:
        print(f"DEBUG: Invalid file format: {file.filename}")
        return jsonify({"error": "Invalid file format. Please upload a CSV file."}), 400

def is_pandas_query(resp):
    pattern = r'global_data\.[a-zA-Z_]+\([^\)]*\)|global_data\[[\'"][^\'"]+[\'"]\]'
    return bool(re.search(pattern, resp))

def generate_query_with_openai(user_input, column_names):
    # Convert column_names to a list if it's not already
    column_list = column_names if isinstance(column_names, list) else column_names.split(", ")
    
    SYSTEM_PROMPT = f"""
    You are an AI assistant analyzing CSV data stored in a pandas DataFrame named 'global_data'.
    Available columns: {', '.join(column_list)}.
    CRITICAL RULES:
    1. Return a SINGLE pandas query starting with 'global_data.'.
    2. Use 'global_data.assign()' for new columns.
    3. ALWAYS use EXACT column names as provided in the available columns list.
    4. Enclose column names in square brackets and double quotes, e.g. global_data["Exact Column Name"].
    5. DO NOT use any float() or other type conversions in the query.
    6. Include '.sort_values()' for ranking or sorting.
    7. End queries with column selection using double brackets, including ALL relevant columns.
    8. Use explicit numerical values: 0.5, 0.25, etc.
    9. Use pd.to_numeric(global_data["Column Name"], errors='coerce') for numeric conversions.
    10. For counting unique values, use .nunique() instead of .unique().count()
    11. For total counts, use .count()
    12. ALWAYS verify column names against the provided list and use exact matches.
    Before writing the query, list the exact column names you will use for:
    1. Patient Incidence
    2. Recruitment Rate
    3. Percentage of sites with no competitor trials
    4. Country
    Then provide the query using these exact column names, ensuring ALL relevant columns are included in the output.
    If the requested data is not present, respond with: "DATA_NOT_PRESENT: <explanation>"
    Always find the user intent and then get the real column name like population is Patient Incidence, recho factor, recho number are Recruitment rate and so on:
    You should always return response in THE GIVEN FORMAT:
    COLUMN NAMES:
    1. Patient Incidence: "Exact Column Name"
    2. Recruitment Rate: "Exact Column Name"
    3. Percentage of sites: "Exact Column Name"
    4. Country: "Exact Column Name"
    PANDAS_QUERY: <single_line_query>
    User Input: {user_input}
    Please never miss to send the data in the format mentioned above. NEVER
    Please note that the global variable name is global_data. global_datapd or anything are wrong and must not be used
    Please read the error and correct the query. The repeat query should not be same as before
    If user inputs that there is something wrong with query, you must update so that it runs
    It will have the error and query, you need to fix that accordingly
    If user does not ask anything, say Hi, hello or anything, respond Hi, how may I help you?
    For COUNT operations, use pd.to_numeric() with errors="coerce" like this  pd.to_numeric(global_data["Country "], errors="coerce")
    check column names {global_data.columns.tolist()} to ensure real column number goes
    Number of countries is 24 if count countries or osmething asked, return 24
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
    print(f"DEBUG: OpenAI response: {resp}")
    
    # Verify and correct column names
    for col in column_list:
        resp = resp.replace(f'["{col.strip()}"]', f'["{col}"]')
    
    return resp

def extract_pandas_query(ai_response):
    
    match = re.search(r'PANDAS_QUERY:\s*(.+)$', ai_response, re.MULTILINE | re.IGNORECASE)
    if match:
        return match.group(1)
    else:
        queries = re.findall(r'PANDAS_QUERY:\s*(global_data\.[^\n]+)', ai_response, re.IGNORECASE)
        return queries if queries else None

def process_query(query):
    query = re.sub(r'global_data\["([^"]+)"\]', r'pd.to_numeric(global_data["\1"], errors="coerce")', query)
    #query = re.sub(r'([-+]?\d*\.\d+|\d+)', r'pd.to_numeric(\1, errors="coerce")', query)
    print("query", query)
    return query

def verify_and_execute_query(query):
    global global_data
    print(f"DEBUG: Executing query: {query}")

    # Replace column names with exact matches from global_data
    for col in global_data.columns:
        query = query.replace(f'["{col.strip()}"]', f'["{col}"]')

    string_expression = process_query(query)
    print(f"DEBUG: String expression: {string_expression}")
    
    
    try:
        result = eval(string_expression, {"global_data": global_data, "pd": pd})
        print(f"DEBUG: Query result type: {type(result)}")
        if isinstance(result, pd.DataFrame):
            return json.dumps({"result": result.to_dict(orient="records")})
        elif isinstance(result, pd.Series):
            return json.dumps({"result": result.to_dict()})
        else:
            return json.dumps({"result": str(result)})
    except Exception as e:
        print(f"DEBUG: Error executing query: {str(e)}")
        return json.dumps({"error": f"Error executing query: {str(e)}"})

@app.route('/chat', methods=['POST'])
def chat():
    global global_data

    if global_data is None:
        return jsonify({"error": "No data uploaded. Please upload a CSV file first."}), 400

    user_input = request.json.get('message')
    if not user_input:
        return jsonify({"error": "No message provided"}), 400

    print(f"DEBUG: Received user input: {user_input}")

    ai_response = generate_query_with_openai(user_input, ", ".join(global_data.columns))
    pandas_query = extract_pandas_query(ai_response)
    
    if pandas_query:
        query_result = verify_and_execute_query(pandas_query)
        return jsonify(json.loads(query_result)), 200
    else:
        # Handle case where no valid pandas query is generated
        return jsonify({"response": ai_response}), 200


if __name__ == '__main__':
    app.run(debug=True)
