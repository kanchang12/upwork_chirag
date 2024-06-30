from flask import Flask, request, render_template, jsonify
import os
import pandas as pd
import openai
import re
import math

# Initialize Flask app
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'

# Set OpenAI API key
openai.api_key = os.getenv('OPENAI_API_KEY')

# Global variable to store the uploaded data
global_data = None

# Route for the index page
@app.route('/')
def index():
    return render_template('index.html')



# Route for file upload
@app.route('/upload', methods=['GET', 'POST'])
def upload_file():
    global global_data

    if 'document' not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files['document']

    if file and file.filename.endswith('.csv'):
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(file_path)

        try:
            global_data = pd.read_csv(file_path)
            return jsonify({"message": "File uploaded successfully", "columns": global_data.columns.tolist()}), 201
        except Exception as e:
            return jsonify({"error": f"Error reading CSV file: {str(e)}"}), 400
    else:
        return jsonify({"error": "Invalid file format. Please upload a CSV file."}), 400

def is_pandas_query(resp):
    pattern = r'global_data\.[a-zA-Z_]+\([^\)]*\)|global_data\[[\'"][^\'"]+[\'"]\]'
    return bool(re.search(pattern, resp))

# Function to generate and verify query
def generate_and_verify_query(user_input, global_data, max_attempts=1):
    if user_input.strip().lower() in ["hi", "hello"]:
        return "Hi, how may I help you?"

    column_names = ", ".join([f"'{col}'" for col in global_data.columns])
    for attempt in range(max_attempts):
        ai_response = generate_query_with_openai(user_input, column_names)
        pandas_query = extract_pandas_query(ai_response)

        if is_pandas_query(pandas_query):
            return pandas_query
        else:
            return ai_response

    return "Failed to generate a valid query. Please try again."

# Function to generate query using OpenAI
def generate_query_with_openai(user_input, column_names):
    SYSTEM_PROMPT = f"""
    # You are an AI assistant analyzing CSV data stored in a pandas DataFrame named 'global_data'.
    # Available columns: {column_names}.

    # CRITICAL RULES:
    # 1. Return a SINGLE pandas query starting with 'global_data.'.
    # 2. Use 'global_data.assign()' for new columns.
    # 3. ALWAYS use EXACT column names as provided in the available columns list.
    # 4. Enclose column names in square brackets and double quotes, e.g. global_data["Exact Column Name"].
    # 5. DO NOT use any float() or other type conversions in the query.
    # 6. Include '.sort_values()' for ranking or sorting.
    # 7. End queries with column selection using double brackets, including ALL relevant columns.
    # 8. Use explicit numerical values: 0.5, 0.25, etc.
    # 9. Use pd.to_numeric(global_data["Column Name"], errors='coerce') for numeric conversions.

    # Before writing the query, list the exact column names you will use for:
    # 1. Patient Incidence
    # 2. Recruitment Rate
    # 3. Percentage of sites with no competitor trials
    # 4. Country

    # Then provide the query using these exact column names, ensuring ALL relevant columns are included in the output.

    # If the requested data is not present, respond with: "DATA_NOT_PRESENT: <explanation>"

    # Format your response as:
    # COLUMN NAMES:
    # 1. Patient Incidence: "Exact Column Name"
    # 2. Recruitment Rate: "Exact Column Name"
    # 3. Percentage of sites: "Exact Column Name"
    # 4. Country: "Exact Column Name"

    # PANDAS_QUERY: <single_line_query>

    # User Input: {user_input}
    Please note that the global variable name is global_data. global_datapd or anything are wrong and must not be used
    Please read the error and correct the query. The repeat query should not be same as before
    If user inputs that there is something wrong with query, you must update so that it runs
    It will have the error and query, you need to fix that accordingly

    If user does not ask anything, say Hi, hello or anything, respond Hi, how may I help you?
    If the question is like this:
    rank the countries in a table based on 50% weight for patient incidence, 25% weight for recruitment rate, and 25% based on % of sites with no competition
    return the query with all the five columns (including the calculated score) and their values
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
    print("resp", resp)
    if is_pandas_query(resp):
        return resp
    else:
        return resp

# Function to extract pandas query from AI response
def extract_pandas_query(ai_response):
    match = re.search(r'PANDAS_QUERY:\s*(.*)', ai_response, re.DOTALL)
    print("match", match)
    if match:
        return match.group(1).strip()
    return None

def process_query(query):
    query = re.sub(r'global_data\["([^"]+)"\]', r'pd.to_numeric(global_data["\1"], errors="coerce")', query)
    query = re.sub(r'([-+]?\d*\.\d+|\d+)', r'pd.to_numeric(\1, errors="coerce")', query)
    return query

def verify_and_execute_query(query):
    global global_data
    
    if bool(re.search(r'\d+(\.\d+)?', query)):
        string_expression = process_query(query)
    else:
        string_expression = query

    result = eval(string_expression, {"global_data": global_data, "pd": pd})
    
    # If the result is a DataFrame, return all columns
    if isinstance(result, pd.DataFrame):
        return result.to_dict(orient='records')
    return result

# Route for chat
@app.route('/chat', methods=['GET', 'POST'])
def chat():
    global global_data

    if global_data is None:
        return jsonify({"error": "No data uploaded. Please upload a CSV file first."}), 400

    user_input = request.json.get('message')
    if not user_input:
        return jsonify({"error": "No message provided"}), 400

    result = generate_and_verify_query(user_input, global_data, max_attempts=1)
    
    if result == "Hi, how may I help you?":
        return jsonify({"Result": result}), 200

    result1 = verify_and_execute_query(result)
    print("result1", result1)
    
    if isinstance(result1, list) and len(result1) > 0 and isinstance(result1[0], dict):
        # This is for DataFrame results
        return jsonify({"Result": result1}), 200
    elif isinstance(result1, (int, float, str)):
        return jsonify({"Result": result1}), 200
    else:
        return jsonify(result1), 200

if __name__ == '__main__':
    app.run(debug=True)
