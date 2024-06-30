


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
@app.route('/upload', methods=['GET', 'POST'])
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
        return jsonify({"response": "Hello! How can I assist you with analyzing the uploaded CSV data?"})

    column_names = ", ".join([f"'{col}'" for col in global_data.columns])
    for attempt in range(max_attempts):
        # Generate query using OpenAI
        ai_response = generate_query_with_openai(user_input, column_names)
        print("AI response:", ai_response)

        # Check if the response is a general response
        if ai_response.lower().startswith('general_response'):
            return jsonify({"response": ai_response.split(':', 1)[1].strip()})

        # Extract the pandas query from the AI response
        pandas_query = extract_pandas_query(ai_response)
        print("Pandas query:", pandas_query)

        if is_pandas_query(pandas_query):
            return pandas_query
        else:
            return jsonify({"response": "I couldn't generate a valid query. Could you please rephrase your question?"})

    return jsonify({"response": "Failed to generate a valid query. Please try again with a more specific question about the data."})

def generate_query_with_openai(user_input, column_names):
    SYSTEM_PROMPT = f"""
# You are an AI assistant analyzing CSV data stored in a pandas DataFrame named 'global_data'.
# Available columns: {column_names}.

# CRITICAL RULES:
# 1. Return a SINGLE pandas query starting with 'global_data.'. The query must be able to return more than one column.
# 2. Use 'global_data.assign()' for new columns.
# 3. ALWAYS use EXACT column names as provided in the available columns list.
# 4. Enclose column names in square brackets and double quotes, e.g. global_data["Exact Column Name"].
# 5. DO NOT use any float() or other type conversions in the query.
# 6. Include '.sort_values()' for ranking or sorting.
# 7. End queries with column selection using double brackets.
# 8. Use explicit numerical values: 0.5, 0.25, etc.
# 9. Use pd.to_numeric(global_data["Column Name"], errors='coerce') for numeric conversions.
# 10. For complex queries, return multiple columns as needed.

# COLUMN MAPPING:
# - Patient Incidence = Population
# - Recruitment Rate = Roche Factor
# - "Percentage of sites with no competitor trials" = use exact column name for this
# - Country  = use exact column name for country

# RESPONSE FORMAT:
# PANDAS_QUERY: <single_line_query>

# IMPORTANT NOTES:
# - The global variable name is global_data. Do not use global_datapd or any other variation.
# - If there's an error, correct the query. The new query should not be the same as before.
# - If user says there's something wrong with the query, update it to make it run correctly.
# - If user doesn't ask anything data-related, respond with "Hi, how may I help you?"
# - For complex queries (e.g., ranking countries based on multiple factors), return all relevant columns.
# - If user asks to share column headers, return a query that includes all columns.
# - If user asks for country and population, return both columns in the query.
# - Always interpret "population" as "Patient Incidence" and "Roche Factor" as "Recruitment Rate".
# - When user asks for multiple fields, include ALL requested columns in the query.

# Please understand what user is trying to say. If it is pandas query, that is asking for specific data, create pandas query as normal
# But if it is general question like hi, return with GENERAL_RESPONSE : <your response>
# if it is not pandas query, always start with the phrase GENERAL_RESPONSE

# can you provide list of countries based on population: this is actually asking for list of counties sorted by population so pandas enquiry, so you have to understand what user is asking for 
# User Input: {user_input}
# Provide a pandas query that accurately addresses the user's request, ensuring all asked-for data is included.
"""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT.format(column_names=column_names)},
        {"role": "user", "content": user_input}
    ]
    response = openai.chat.completions.create(
        model="gpt-3.5-turbo-16k",
        messages=messages,
        temperature=0,
        max_tokens=650
    )
    resp = response.choices[0].message.content
    print("response: ", resp)
    
    print("AI Return: ", resp)
    if resp.lower().startswith('General_response'.lower()):
        result1 = resp
    else:
        return resp


# Function to extract pandas query from AI response
def extract_pandas_query(ai_response):
    match = re.search(r'PANDAS_QUERY:\s*(.*)', ai_response, re.DOTALL)
    if match:
        return match.group(1)
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
    

@app.route('/chat', methods=['GET', 'POST'])
def chat():
    global global_data

    if global_data is None:
        return jsonify({"error": "No data uploaded. Please upload a CSV file first."}), 400

    user_input = request.json.get('message')
    if not user_input:
        return jsonify({"error": "No message provided"}), 400

    result = generate_and_verify_query(user_input, global_data, max_attempts=1)
    
    # Check if result is already a JSON response
    if isinstance(result, tuple) and len(result) == 2 and isinstance(result[0], dict):
        return result

    # If it's a pandas query, process it as before
    result1 = verify_and_execute_query(result)
    print("result", result1)
    
    # Rest of the function remains the same...
    # Check if result1 is a pandas DataFrame
    if isinstance(result1, pd.DataFrame):
        # Check for NaN values in any column
        if result1.isnull().any().any():
            # Drop columns with NaN values
            result1 = result1.dropna(axis=1)
            RESULT1_STR = result1.to_string(index=False)
        
            return jsonify(RESULT1_STR)
            #return render_template('index.html', table=result1.to_html())
        else:
            # Render single value to HTML template
            RESULT1_STR = result1.to_string(index=False)
            return jsonify(RESULT1_STR)
            #return render_template('index.html', value=result1)
    elif isinstance(result1, list):

        return jsonify(''.join(', '.join(map(str, row)) for row in result1))
    else:
        return jsonify(result1)
    """
    if isinstance(result1, pd.DataFrame) or isinstance(result1, pd.Series):
        result1 = result1.dropna(axis=1).to_dict(orient='records')
        first_item = result1[0]  # Get the first dictionary in the list
        first_key = next(iter(first_item))
        values_list = [item[first_key] for item in result1]
        print(values_list)
        return jsonify({"Result": values_list}), 200
    elif isinstance(result1, (int, float, str)):
            return jsonify(format_result(result1))
    else:
            return jsonify(result1)
"""

if __name__ == '__main__':
    app.run(debug=True)
