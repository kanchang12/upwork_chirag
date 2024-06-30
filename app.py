

from flask import Flask, request, render_template, jsonify
import os
import pandas as pd
import openai
import re
import math

# Initialize Flask app
#app = Flask(__name__)
openai.api_key = os.getenv('OPENAI_API_KEY')

# Initialize Flask app
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024 

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
        print("'document' not in request.files")
        return jsonify({"error": "No file part in the request"}), 400

    file = request.files['document']

    if file.filename == '':
        print("File name is empty")
        return jsonify({"error": "No file selected"}), 400

    if file and file.filename.endswith('.csv'):
        try:
            filename = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filename)
            global_data = pd.read_csv(filename)
            print("\n\n\nglobal_data at load", global_data)
            print(f"File saved and loaded: {filename}")
            return jsonify({
                "message": "File uploaded successfully",
                "columns": global_data.columns.tolist()
            }), 200
        except Exception as e:
            print(f"Error processing file: {str(e)}")
            return jsonify({"error": f"Error processing file: {str(e)}"}), 500
    else:
        print(f"Invalid file format: {file.filename}")
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
        print("Second value ", pandas_query)
        print("First value ", ai_response)

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
    # Always find the user intent and then get the real column name like population is Patience Incident, recho facor, recho number are Recruitment rate and so on:
    # You should always return response in THE GIVEN FORMATCOLUMN NAMES:
    # 1. Patient Incidence: "Exact Column Name"
    # 2. Recruitment Rate: "Exact Column Name"
    # 3. Percentage of sites: "Exact Column Name"
    # 4. Country: "Exact Column Name"
    # PANDAS_QUERY: <single_line_query>
    #Till this format
    # User Input: {user_input}
    # Please never miss to send the data in the format mentioned above. NEVER
    Please note that the global variable name is global_data. global_datapd or anything are wrong and must not be used
    Please read the error and correct the query. The repeat query should not be same as before
    If user inputs that there is something wrong with query, you must update so that it runs
    It will have the error and query, you need to fix that accordingly
    If user does not ask anything, say Hi, hello or anything, respond Hi, how may I help you?
    If the question is like this:
    
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
    print("user input", user_input)
    print("response", response)
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
        return match.group(1)
    return ai_response

def process_query(query):
    #query = re.sub(r'global_data\["([^"]+)"\]', r'pd.to_numeric(global_data["\1"], errors="coerce")', query)
    #query = re.sub(r'([-+]?\d*\.\d+|\d+)', r'pd.to_numeric(\1, errors="coerce")', query)
    print("query", query)
    return query

def verify_and_execute_query(query):
    global global_data

    if is_pandas_query(query):
        string_expression = process_query(query)
        print("string_expression", string_expression)
        print("\n\n\nglobal_data", global_data)
        try:
            result = eval(string_expression, {"global_data": global_data, "pd": pd})
            print("result", result)
            if isinstance(result, pd.DataFrame):
                return result.to_json(orient="records")
            return result
        except Exception as e:
            return {"error": f"Error executing query: {str(e)}"}
    else:
        try:
            response = openai.chat.completions.create(
                model="gpt-3.5-turbo-16k",
                messages=[
                    {"role": "user", "content": query}
                ],
                temperature=0,
                max_tokens=450
            )
            ai_response = response.choices[0].message.content
            return {"response": ai_response}
        except Exception as e:
            return {"error": f"Error processing AI response: {str(e)}"}

# Route for chat
@app.route('/chat', methods=['GET', 'POST'])
def chat():
    global global_data

    if global_data is None:
        return jsonify({"error": "No data uploaded. Please upload a CSV file first."}), 400

    user_input = request.json.get('message')
    if not user_input:
        return jsonify({"error": "No message provided"}), 400

    # Example function call to generate and verify query
    result = generate_and_verify_query(user_input, global_data, max_attempts=1)
    
    if result == "Hi, how may I help you?":
        return jsonify({"result": result}), 200

    # Example function call to verify and execute query
    result1 = verify_and_execute_query(result)
    print("result1", result1)
    
    # Check result type and format accordingly
    if isinstance(result1, list) and len(result1) > 0 and isinstance(result1[0], dict):
        # This is for DataFrame results
        return jsonify({"result": result1}), 200
    elif isinstance(result1, (int, float, str)):
        # This is for single result values
        return jsonify({"result": result1}), 200
    else:
        # Handle other types of results as needed
        return jsonify({"result": "Unknown result type"}), 200

if __name__ == '__main__':
    app.run(debug=True)
