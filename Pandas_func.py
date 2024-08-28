import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image

# Upload the CSV file
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
Pandas = Image.open("Pandas.jpg")
st.image(Pandas, use_column_width=True)
st.title("Hands-On Pandas Functions in Streamlit")

# Check if a file is uploaded
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    
    def create_boxplot(df):
        fig, ax = plt.subplots()  # Create a figure and axes
        df.boxplot(ax=ax)         # Plot on the axes
        return fig                # Return the figure
    
    functions_dict = {
        "read_csv": {
            "description": "Reads a CSV file into a DataFrame.",
            "syntax": "pd.read_csv(filepath_or_buffer)",
            "example": "df = pd.read_csv('file.csv')",
            "result": lambda df: df.head()
        },
        "describe": {
            "description": "Displays a concise statistics of the DataFrame.",
            "syntax": "df.describe()",
            "example": "df.describe()",
            "result": lambda df: df.describe()
        },
        "isnull_sum": {
            "description": "Returns the number of missing values in each column.",
            "syntax": "df.isnull().sum()",
            "example": "df.isnull().sum()",
            "result": lambda df: df.isnull().sum()
        },
        "value_counts": {
            "description": "Returns a Series containing counts of unique values.",
            "syntax": "df['column'].value_counts()",
            "example": "df['column'].value_counts()",
            "result": lambda df, col: df[col].value_counts() if col else None
        },
        "unique": {
            "description": "Returns unique values of the selected column.",
            "syntax": "df['column'].unique()",
            "example": "df['column'].unique()",
            "result": lambda df, col: df[col].unique() if col else None
        },
        "nunique": {
            "description": "Returns the number of unique values per column.",
            "syntax": "df.nunique()",
            "example": "df.nunique()",
            "result": lambda df: df.nunique()
        },
        "shape": {
            "description": "Returns the shape of the DataFrame.",
            "syntax": "df.shape",
            "example": "df.shape",
            "result": lambda df: df.shape
        },
        "crosstab": {
            "description": "Computes a cross-tabulation of two columns.",
            "syntax": "pd.crosstab(df['col1'], df['col2'])",
            "example": "pd.crosstab(df['col1'], df['col2'])",
            "result": lambda df, col1, col2: pd.crosstab(df[col1], df[col2]) if col1 and col2 else None
        },
        "pivot_table": {
            "description": "Creates a pivot table as a DataFrame.",
            "syntax": "pd.pivot_table(df, index='col1', columns='col2', values='col3', aggfunc='mean')",
            "example": "pd.pivot_table(df, index='col1', columns='col2', values='col3', aggfunc='mean')",
            "result": lambda df, index, columns, values: pd.pivot_table(df, index=index, columns=columns, values=values, aggfunc='mean') if index and columns and values else None
        },
        "groupby": {
            "description": "Groups DataFrame using a mapper or by a Series of columns.",
            "syntax": "df.groupby('col').agg({'col_to_agg':'mean'})",
            "example": "df.groupby('col').agg({'col_to_agg':'mean'})",
            "result": lambda df, group_col, agg_col: df.groupby(group_col)[agg_col].mean() if group_col and agg_col else None
        },
        "loc": {
            "description": "Access a group of rows and columns by labels.",
            "syntax": "df.loc[start:end, ['col1', 'col2']]",
            "example": "df.loc[0:5, ['col1', 'col2']]",
            "result": lambda df: df.loc[0:5, :]
        },
        "drop": {
            "description": "Drops specified labels from rows or columns.",
            "syntax": "df.drop(columns=['col_to_drop'])",
            "example": "df.drop(columns=['col_to_drop'])",
            "result": lambda df, col_to_drop: df.drop(columns=[col_to_drop]) if col_to_drop else None
        },
        "apply": {
            "description": "Applies a function along the axis of the DataFrame.",
            "syntax": "df['col'].apply(lambda x: x * 2)",
            "example": "df['col'].apply(lambda x: x * 2)",
            "result": lambda df, col: df[col].apply(lambda x: x * 2) if col else None
        },
        "duplicated": {
            "description": "Returns boolean Series denoting duplicate rows.",
            "syntax": "df.duplicated()",
            "example": "df[df.duplicated()]",
            "result": lambda df: df[df.duplicated()]
        },
        "boxplot": {
            "description": "Gives the Statistical Measures of the data.",
            "syntax": "df.boxplot()",
            "example": "df.boxplot()",
            "result": create_boxplot
        },
        "merge": {
            "description": "Merges two DataFrames based on a common column or index.",
            "syntax": "pd.merge(df1, df2, on='key')",
            "example": "pd.merge(df1, df2, on='common_column')",
            "result": lambda df1, df2, key: pd.merge(df1, df2, on=key) if key else None
        },
        "concat": {
            "description": "Concatenates two or more DataFrames along a particular axis.",
            "syntax": "pd.concat([df1, df2], axis=0)",
            "example": "pd.concat([df1, df2], axis=0)",
            "result": lambda df1, df2, axis: pd.concat([df1, df2], axis=axis) if df2 is not None else None
        },
        "mean": {
            "description": "Returns the mean of the DataFrame's numerical columns.",
            "syntax": "df.mean()",
            "example": "df.mean()",
            "result": lambda df: df.mean()
        },
        "median": {
            "description": "Returns the median of the DataFrame's numerical columns.",
            "syntax": "df.median()",
            "example": "df.median()",
            "result": lambda df: df.median()
        },
        "var": {
            "description": "Returns the variance of the DataFrame's numerical columns.",
            "syntax": "df.var()",
            "example": "df.var()",
            "result": lambda df: df.var()
        },
        "std": {
            "description": "Returns the standard deviation of the DataFrame's numerical columns.",
            "syntax": "df.std()",
            "example": "df.std()",
            "result": lambda df: df.std()
        },
        "to_datetime": {
            "description": "Converts a column to datetime.",
            "syntax": "pd.to_datetime(df['column'])",
            "example": "pd.to_datetime(df['date_column'])",
            "result": lambda df, col: pd.to_datetime(df[col]) if col else None
        },
        "astype": {
            "description": "Casts a DataFrame column to a different data type.",
            "syntax": "df['column'].astype(dtype)",
            "example": "df['column'].astype('float')",
            "result": lambda df, col, dtype: df[col].astype(dtype) if col and dtype else None
        }
    }

    # UI for selecting and displaying function details and results
    selected_function = st.selectbox("Select a pandas function", list(functions_dict.keys()))

    st.write("### Description:")
    st.write(functions_dict[selected_function]["description"])

    st.write("### Syntax:")
    st.code(functions_dict[selected_function]["syntax"])

    st.write("### Example:")
    st.code(functions_dict[selected_function]["example"])

    # Depending on the selected function, display the result
    if selected_function in ["value_counts", "unique", "apply"]:
        column = st.selectbox("Select column", df.columns)
        st.write("### Result:")
        st.write(functions_dict[selected_function]["result"](df, column))
    elif selected_function == "crosstab":
        col1 = st.selectbox("Select first column", df.columns, key="col1")
        col2 = st.selectbox("Select second column", df.columns, key="col2")
        if col1 and col2:
            st.write("### Result:")
            st.write(functions_dict[selected_function]["result"](df, col1, col2))
    elif selected_function == "merge":
        uploaded_file2 = st.file_uploader("Upload another CSV file", type="csv", key="2")
        if uploaded_file2 is not None:
            df2 = pd.read_csv(uploaded_file2)
            key = st.text_input("Enter the key column for merging")
            if key:
                st.write("### Result:")
                st.write(functions_dict[selected_function]["result"](df, df2, key))
    elif selected_function == "concat":
        uploaded_file2 = st.file_uploader("Upload another CSV file", type="csv", key="2")
        if uploaded_file2 is not None:
            df2 = pd.read_csv(uploaded_file2)
            axis = st.radio("Select axis for concatenation", options=[0, 1])
            st.write("### Result:")
            st.write(functions_dict[selected_function]["result"](df, df2, axis))
    elif selected_function == "pivot_table":
        index = st.text_input("Enter index column")
        columns = st.text_input("Enter columns")
        values = st.text_input("Enter values")
        if index and columns and values:
            st.write("### Result:")
            st.write(functions_dict[selected_function]["result"](df, index, columns, values))
    elif selected_function == "groupby":
        group_col = st.text_input("Enter the column to group by")
        agg_col = st.text_input("Enter the column to aggregate")
        if group_col and agg_col:
            st.write("### Result:")
            st.write(functions_dict[selected_function]["result"](df, group_col, agg_col))
    elif selected_function in ["loc", "drop", "apply"]:
        if selected_function == "loc":
            st.write("### Result:")
            st.write(functions_dict[selected_function]["result"](df))
        elif selected_function == "drop":
            col_to_drop = st.selectbox("Select column to drop", df.columns)
            st.write("### Result:")
            st.write(functions_dict[selected_function]["result"](df, col_to_drop))
        elif selected_function == "apply":
            col = st.selectbox("Select column to apply function", df.columns)
            st.write("### Result:")
            st.write(functions_dict[selected_function]["result"](df, col))
    elif selected_function == "boxplot":
        st.write("### Result:")
        st.pyplot(functions_dict[selected_function]["result"](df))
    elif selected_function in ["to_datetime", "astype"]:
        col = st.selectbox("Select column", df.columns)
        if selected_function == "astype":
            dtype = st.text_input("Enter data type")
            if dtype:
                st.write("### Result:")
                st.write(functions_dict[selected_function]["result"](df, col, dtype))
        else:
            st.write("### Result:")
            st.write(functions_dict[selected_function]["result"](df, col))
    else:
        st.write("### Result:")
        st.write(functions_dict[selected_function]["result"](df))
