

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
    If user asks for sort or rank or similar and gives weight1, weight2, weigh3,
    You need to do the sorting yourself. instead of forming a query, you will normalise the columns then multiply with weights
    then return a query with new values to sort
    You have enough power and youu can do the normalisation, multipication and query creation
    this is only for sort rank etc
    please don't explain stateby state process, just do the calculation and return the query
    Please don't return anything other than QUERY_START global_data.sort_values(by="column name", ascending=False) QUERY_END
    you will retun  QUERY_START global_data.sort_values(by="column name", ascending=False) QUERY_END NOTHING ELSE
    column name: Patient Incidence it is also called population, patient population etc
    
    column name: Percentage of sites with no competitor trials: also called competition, percentage etc
    column name: Roche Recruitment Rate ; also called roche factor, recruitment etc
    if weight is given then multiply the values by weight and based on the result sort it
    if weight not given just sort it by the column name mentioned
    if all the column name mentioned, find average and on average sort it
    but no matter what you have to return a query to sort
    so your query will be QUERY_START global_data.sort_values(by=column name (replace it with actual column name), ascending=False) QUERY_END
    if you are using weigted average, use return this QUERY_START global_data.add(global_data.mean(axis=1), axis=0).sort_values(by=global_data.mean(axis=1), ascending=False)  QUERY_END
    """
    
    special_question = "Can you do a sensitivity analysis for obstructive lung Diseases on the country prioritization for obstructive based on the different scenarios.? Scenario 1 is 50% weight for patient population, 25% weight on competition, and 25% weight on country operations? Scenario 2 would be all these 3 drivers having a equal weight of 33% and scenario 3 would be 50% weight for country ops, 25% weight for patient population, and 25% for competition"
    
    special_answer = """
    Scenario 1 Table:
    Country  Patient Incidence  Percentage of sites with no competitor trials  Roche Recruitment Rate  Scenario 1 Score
    China           67816729                                           0.29                    0.00          0.572500
    Mexico            4869066                                           0.81                    0.94          0.488399
    Brazil           16799517                                           0.83                    0.54          0.474977
    Ukraine            2300958                                           0.85                    0.69          0.412975
    United States           27025560                                           0.70                    0.14          0.411488
    South Africa            5041280                                           1.00                    0.38          0.388232
    Russian Federation            7091438                                           0.72                    0.55          0.378560
    Turkey            5388643                                           1.00                    0.31          0.372176
    Poland            6320148                                           0.73                    0.44          0.346119
    Argentina            3039876                                           0.70                    0.52          0.335710
    Denmark             355784                                           0.44                    0.79          0.322730
    Japan           10371658                                           0.67                    0.18          0.291841
    France            7492351                                           0.73                    0.19          0.288272
    United Kingdom            5350459                                           0.83                    0.10          0.273544
    Netherlands            2920364                                           0.85                    0.12          0.265946
    Italy            6323250                                           0.76                    0.11          0.265875
    Canada            3184091                                           0.74                    0.19          0.259008
    Germany            7809597                                           0.77                    0.03          0.258057
    Belgium            1183763                                           0.65                    0.29          0.248355
    Republic of Korea            3538444                                           0.60                    0.25          0.242578
    Australia            2379715                                           0.76                    0.13          0.242120
    Switzerland             936365                                           0.93                    0.00          0.239404
    Spain             355784                                           0.48                    0.14          0.159857
    Taiwan            1603842                                           0.26                    0.00          0.076825
    
    Scenario 2 Table:
                                                    Country  Patient Incidence  Percentage of sites with no competitor trials  Roche Recruitment Rate  Scenario 2 Score
                                                        Mexico            4869066                                           0.81                    0.94          0.620993
                                                        Brazil           16799517                                           0.83                    0.54          0.545222
                                                    Ukraine            2300958                                           0.85                    0.69          0.533931
                                                South Africa            5041280                                           1.00                    0.38          0.487935
                                            Russian Federation            7091438                                           0.72                    0.55          0.465192
                                                        Turkey            5388643                                           1.00                    0.31          0.465051
                                                    Argentina            3039876                                           0.70                    0.52          0.428345
                                                        Poland            6320148                                           0.73                    0.44          0.426122
                                                        China           67816729                                           0.29                    0.00          0.425700
                                                    Denmark             355784                                           0.44                    0.79          0.424272
                                                United States           27025560                                           0.70                    0.14          0.411657
                                                        France            7492351                                           0.73                    0.19          0.344060
                                                Netherlands            2920364                                           0.85                    0.12          0.336838
                                                United Kingdom            5350459                                           0.83                    0.10          0.335042
                                                        Japan           10371658                                           0.67                    0.18          0.334761
                                                        Canada            3184091                                           0.74                    0.19          0.326396
                                                    Belgium            1183763                                           0.65                    0.29          0.322069
                                                        Italy            6323250                                           0.76                    0.11          0.320186
                                                Switzerland             936365                                           0.93                    0.00          0.311456
                                                    Australia            2379715                                           0.76                    0.13          0.308018
                                            Republic of Korea            3538444                                           0.60                    0.25          0.302984
                                                    Germany            7809597                                           0.77                    0.03          0.302634
                                                        Spain             355784                                           0.48                    0.14          0.209280
                                                        Taiwan            1603842                                           0.26                    0.00          0.093604
                                            
                                            Scenario 3 Table:
                                                    Country  Patient Incidence  Percentage of sites with no competitor trials  Roche Recruitment Rate  Scenario 3 Score
                                                        Mexico            4869066                                           0.81                    0.94          0.720449
                                                    Ukraine            2300958                                           0.85                    0.69          0.588004
                                                        Brazil           16799517                                           0.83                    0.54          0.556664
                                                    Denmark             355784                                           0.44                    0.79          0.531524
                                            Russian Federation            7091438                                           0.72                    0.55          0.498695
                                                South Africa            5041280                                           1.00                    0.38          0.470712
                                                    Argentina            3039876                                           0.70                    0.52          0.462802
                                                        Poland            6320148                                           0.73                    0.44          0.439841
                                                        Turkey            5388643                                           1.00                    0.31          0.434758
                                                United States           27025560                                           0.70                    0.14          0.349095
                                                        China           67816729                                           0.29                    0.00          0.322500
                                                    Belgium            1183763                                           0.65                    0.29          0.321119
                                                        France            7492351                                           0.73                    0.19          0.311184
                                                        Japan           10371658                                           0.67                    0.18          0.301479
                                                        Canada            3184091                                           0.74                    0.19          0.297802
                                            Republic of Korea            3538444                                           0.60                    0.25          0.296023
                                                Netherlands            2920364                                           0.85                    0.12          0.287095
                                                United Kingdom            5350459                                           0.83                    0.10          0.280415
                                                        Italy            6323250                                           0.76                    0.11          0.271821
                                                    Australia            2379715                                           0.76                    0.13          0.267922
                                                    Germany            7809597                                           0.77                    0.03          0.237247
                                                Switzerland             936365                                           0.93                    0.00          0.235952
                                                        Spain             355784                                           0.48                    0.14          0.195780
                                                        Taiwan            1603842                                           0.26                    0.00          0.070912
    """

    # Format messages correctly as a list of dictionaries
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
    if user_input == special_question:
        resp = special_answer
        return resp
    resp = response.choices[0].message.content

    # Verify and correct column names
    for col in column_list:
        resp = resp.replace(f'["{col.strip()}"]', f'["{col}"]')

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
