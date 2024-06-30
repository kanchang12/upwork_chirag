from flask import Flask, request, render_template, jsonify
import os
import pandas as pd
import openai

import re
import math

# Load environment variables from .env file


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

@app.route('/second')
def second_page():
    return render_template('second.html')

# Route for file upload
@app.route('/upload', methods=['POST'])
def upload_file():
    global global_data
    global global_datapd

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
    # Define regex pattern to match valid pandas query structures
    pattern = r'global_data\.[a-zA-Z_]+\([^\)]*\)|global_data\[[\'"][^\'"]+[\'"]\]'
    
    # Use re.search to find if the pattern matches the response
    return bool(re.search(pattern, resp))


# Function to generate and verify query
def generate_and_verify_query(user_input, global_data, max_attempts=1):
    # Check if user input is a greeting
    if user_input.strip().lower() in ["hi", "hello"]:
        return render_template('index.html', response=f"Hi, how may I help you?")

    column_names = ", ".join([f"'{col}'" for col in global_data.columns])
    for attempt in range(max_attempts):
        # Generate query using OpenAI
        ai_response = generate_query_with_openai(user_input, column_names)

        # Extract the pandas query from the AI response
        pandas_query = extract_pandas_query(ai_response)

        if is_pandas_query(pandas_query):
            return pandas_query
        else:
            return render_template('index.html', response=ai_response)

    return render_template('index.html', response="Failed to generate a valid query. Please try again.")

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
    # 7. End queries with column selection using double brackets, including ALL requested columns.
    # 8. Use explicit numerical values: 0.5, 0.25, etc.
    # 9. Use pd.to_numeric(global_data["Column Name"], errors='coerce') for numeric conversions.
    # 10. When user asks for multiple columns or values, ALWAYS include all requested columns in the final selection.

    # If the requested data is not present, respond with: "DATA_NOT_PRESENT: <explanation>"

    # Format your response as:
    # PANDAS_QUERY: <single_line_query>

    # User Input: {user_input}
    Please note that the global variable name is global_data. global_datapd or anything are wrong and must not be used.
    Please read the error and correct the query. The repeat query should not be same as before.
    If user inputs that there is something wrong with query, you must update so that it runs.
    It will have the error and query, you need to fix that accordingly.

    If user does not ask anything, say Hi, hello or anything, respond Hi, how may I help you?
    If the question asks for multiple columns or values, ensure ALL requested columns are included in the final query selection.
    """
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT.format(column_names=column_names)},
        {"role": "user", "content": user_input}
    ]
    response = openai.chat.completions.create(
        model="gpt-3.5-turbo-16k",
        messages=messages,
        temperature=0,
        max_tokens=450
    )
    resp = response.choices[0].message.content
    if is_pandas_query(resp):
        return resp
    else:
        return render_template('index.html', response=resp)


# Function to extract pandas query from AI response
def extract_pandas_query(ai_response):
    match = re.search(r'PANDAS_QUERY:\s*(.*)', ai_response, re.DOTALL)
    if match:
        return match.group(1).strip()
    return None

def process_query(query):
    # Replace column references with pd.to_numeric to ensure proper conversion
    query = re.sub(r'global_data\["([^"]+)"\]', r'pd.to_numeric(global_data["\1"], errors="coerce")', query)

    # Replace numeric constants with pd.to_numeric to ensure consistency in operations
    query = re.sub(r'([-+]?\d*\.\d+|\d+)', r'pd.to_numeric(\1, errors="coerce")', query)

    return query

def format_result(result):
    if isinstance(result, dict) and "Result" in result:
        pass

    elif isinstance(result, pd.Series):
        # Check if the result is a Series (e.g., country scores)
        formatted_result = []
        for country, score in result.items():
            if pd.notna(score):  # Exclude NaN values
                formatted_result.append({"Country": country, "Score": score})
        return formatted_result

    elif isinstance(result, (int, float, str)):
        return result  # Return directly if it's a single value (e.g., country name)

    else:
        return "Unsupported result format"
    

# Function to verify and execute the query
def verify_and_execute_query(query):
    global global_data
    global_datapd = global_data.copy()  # Create a copy for processing
    
    
    if bool(re.search(r'\d+(\.\d+)?', query)):
        string_expression = process_query(query)
    else:
        string_expression = query

    result = eval(string_expression, {"global_data": global_data, "pd": pd})
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
    result1 = verify_and_execute_query(result)
    print(result1)
    
    if isinstance(result1, pd.DataFrame):
        # Convert DataFrame to list of dicts, including column headers
        result_list = result1.to_dict(orient='records')
        return jsonify({"Result": result_list, "Columns": result1.columns.tolist()}), 200
    elif isinstance(result1, pd.Series):
        # Convert Series to list of dicts, including index as a column
        result_list = [{"Index": idx, "Value": val} for idx, val in result1.items()]
        return jsonify({"Result": result_list, "Columns": ["Index", "Value"]}), 200
    elif isinstance(result1, (list, np.ndarray)):
        # Handle list or array
        return jsonify({"Result": result1.tolist() if isinstance(result1, np.ndarray) else result1}), 200
    elif isinstance(result1, (int, float, str)):
        return jsonify({"Result": result1}), 200
    else:
        return jsonify({"Result": str(result1)}), 200


if __name__ == '__main__':
    app.run(debug=True)
