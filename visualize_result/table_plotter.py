import pandas as pd

def print_result_table(df: pd.DataFrame, category_columns: list = ['pe', 'evaluator', 'fewshot']):
    # Define experiment columns for grouping with iteration information
    experiment_cols_with_iterations = category_columns

    # Define the columns related to the specific iterations and simplified metric names
    iteration_columns = {
        'Best Similarity Iteration (Mean ± Std)': ['Max Similarity Iteration', 'Max Normalized Similarity', 'Diversity at Max Normalized Similarity'],
        'Final Iteration (Mean ± Std)': ['Final Normalized Similarity', 'Final Normalized Diversity']
    }

    # Map for simplified column names
    simplified_columns = {
        'Max Similarity Iteration': 'Iteration',
        'Max Normalized Similarity': 'Similarity',
        'Diversity at Max Normalized Similarity': 'Diversity',
        'Final Normalized Similarity': 'Similarity',
        'Final Normalized Diversity': 'Diversity'
    }

    # Flatten the dictionary to use it in groupby aggregation
    flat_columns = [item for sublist in iteration_columns.values() for item in sublist]

    # Aggregation function to format mean ± std
    def mean_std_formatter(series):
        mean = series.mean()
        std = series.std()
        return f"{mean:.3f} ± {std:.3f}"

    # Grouped result with mean ± std
    grouped_result = df.groupby(experiment_cols_with_iterations)[flat_columns].agg(mean_std_formatter)

    # Rename columns based on the simplified column names without the "Summary" level
    grouped_result = grouped_result.rename(columns=simplified_columns)

    # Create a MultiIndex for better readability by restructuring the DataFrame, omitting "Summary"
    grouped_result.columns = pd.MultiIndex.from_tuples(
        [(key, simplified_columns[col]) for key, cols in iteration_columns.items() for col in cols],
        names=["Iteration Type", "Metric"]
    )

    # Define custom order for 'pe' column and sort by it
    custom_order = ['io', 'cot', 'tot', 'got']

    # consider tot-(postfix), got-(postfix) in sorting
    # depareated: grouped_result = grouped_result.reindex(custom_order, level=0)
    # Define custom sorting function for custom_order with postfix consideration
    def custom_sort_function(index_value):
        base_value = index_value.split('-')[0]  # Remove postfix
        base_priority = custom_order.index(base_value) if base_value in custom_order else len(custom_order)
        return (base_priority, index_value)  # Sort by base priority and original value for postfix ordering

    # Reindex with custom sorting
    sorted_index = sorted(grouped_result.index, key=custom_sort_function)

    # Apply sorted index to the DataFrame
    grouped_result = grouped_result.loc[sorted_index]

    # Style the DataFrame for cleaner display in Jupyter with center alignment
    styled_grouped_result = grouped_result.style.set_properties(**{
        'text-align': 'center',
        'border': '1px solid black'
    }).set_table_styles([
        {
            'selector': 'caption',
            'props': [('font-size', '16px'), ('text-align', 'center'), ('font-weight', 'bold')]
        },
        {
            'selector': 'th',
            'props': [('text-align', 'center')]
        },
        {
            'selector': 'td',
            'props': [('text-align', 'center')]
        }
    ])

    # Display styled table
    return styled_grouped_result