from flask import Flask, render_template, request
import pandas as pd
import plotly.express as px
import plotly.io as pio
import plotly.graph_objects as go
import hiplot as hip
import tempfile


app = Flask(__name__)

# Load the dataset
df = pd.read_csv('WA_Fn-UseC_-Telco-Customer-Churn.csv')
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df_n = pd.get_dummies(df.drop(columns=['customerID', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies']), drop_first=True)
corr_matrix = df_n.corr()

@app.route('/corr_heat_py')
def correlation_heatmap():
    fig = px.imshow(corr_matrix,
                    labels=dict(color="Correlation"),
                    x=corr_matrix.columns,
                    y=corr_matrix.columns,
                    color_continuous_scale='Viridis')

    fig.update_layout(
        title='Correlation Matrix Heatmap',
        xaxis_nticks=36,
        yaxis_nticks=36,
        width=1200,
        height=800
    )

    fig.update_traces(
        hovertemplate='<b>%{x}</b> vs <b>%{y}</b><br>Correlation: %{z:.2f}<extra></extra>',
        customdata=[(x, y) for x in corr_matrix.columns for y in corr_matrix.columns]
    )

    fig.update_layout(clickmode='event+select')

    heatmap_html = pio.to_html(fig, full_html=False)

    return render_template('corr_heat_py.html', heatmap_html=heatmap_html)

@app.route('/scatter_plot')
def scatter_plot():
    x = request.args.get('x')
    y = request.args.get('y')

    # Map dummy variable names back to original column names
    dummy_columns = {col: col.split('_')[0] for col in df_n.columns if '_' in col}
    x_original = dummy_columns.get(x, x)
    y_original = dummy_columns.get(y, y)

    if x_original in df.columns and y_original in df.columns:
        # Create a copy of the DataFrame to avoid modifying the original
        df_copy = df.copy()

        # Ensure the columns are numeric
        if not pd.api.types.is_numeric_dtype(df_copy[x_original]):
            df_copy[x_original] = pd.factorize(df_copy[x_original])[0]
        if not pd.api.types.is_numeric_dtype(df_copy[y_original]):
            df_copy[y_original] = pd.factorize(df_copy[y_original])[0]

        fig = px.scatter(df_copy, x=x_original, y=y_original, trendline='ols')
        fig.update_layout(title=f'Scatter Plot of {x_original} vs {y_original}')
        scatter_html = pio.to_html(fig, full_html=False)
        return render_template('scatter_plot.html', scatter_html=scatter_html)
    else:
        return "Invalid parameters", 400

@app.route('/hiplot')
def hiplot():
    # Preprocess data for Hiplot
    data_filtered = df[['PaymentMethod','PaperlessBilling', 'InternetService','MonthlyCharges', 'TotalCharges', 'Churn']]
    data_filtered['Churn'] = data_filtered['Churn'].map({'Yes': 1, 'No': 0})

    # Create Hiplot experiment
    hip_data = hip.Experiment.from_dataframe(data_filtered)

    # Save Hiplot visualization to a temporary HTML file
    with tempfile.NamedTemporaryFile(delete=False, suffix='.html') as tmp:
        hip_data.to_html(tmp.name)
        tmp.seek(0)
        hip_html = tmp.read().decode('utf-8')

    return render_template('hiplot.html', hip_html=hip_html)

@app.route('/pie')
def pie_chart():
    # tableau
    return render_template('pie.html')

@app.route('/corr_map')
def corr_map():
    # tableau:
    return render_template('corr_map.html')

@app.route('/corr_churn')
def corr_churn():
    # tableau: single var
    return render_template('corr_churn.html')

@app.route('/sunburst')
def sunburst():
    # Count the occurrences of 'Yes' and 'No' in the 'Churn' column
    df_counts = df.groupby(['Churn', 'Contract', 'PaymentMethod', 'InternetService']).size().reset_index(name='Count')

    fig = px.sunburst(df_counts,
                      path=['Churn', 'Contract', 'PaymentMethod', 'InternetService'],
                      values='Count',
                      color='Churn',
                      color_continuous_scale='RdBu',
                      title='Telco Customer Churn Sunburst Chart',
                      labels={'Churn': 'Churn', 'Contract': 'Contract', 'PaymentMethod': 'Payment Method', 'InternetService': 'Internet Service'})

    fig.update_layout(
        transition_duration=500,  # Add animation duration
        width=1200,  # Set the width of the chart
        height=800   # Set the height of the chart
    )

    # fig.update_traces(
    #     hovertemplate='%{meta[0]}<extra></extra>',
    #     meta=['Churn', 'Contract', 'PaymentMethod', 'InternetService']
    # )

    sunburst_html = pio.to_html(fig, full_html=False)

    return render_template('sunburst.html', sunburst_html=sunburst_html)

@app.route('/')
def tree():
    columns_hierarchy = {
        'Customer': {
            'Demographics': ['Gender', 'SeniorCitizen', 'Partner', 'Dependents'],
            'Services': ['PhoneService', 'MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies'],
            'Account': ['Contract', 'PaperlessBilling', 'PaymentMethod', 'MonthlyCharges', 'TotalCharges']
        }
    }

    def build_tree(data, parent_name=''):
        labels = []
        parents = []
        for key, value in data.items():
            labels.append(key)
            parents.append(parent_name)
            if isinstance(value, dict):
                child_labels, child_parents = build_tree(value, key)
                labels.extend(child_labels)
                parents.extend(child_parents)
            else:
                labels.extend(value)
                parents.extend([key] * len(value))
        return labels, parents

    labels, parents = build_tree(columns_hierarchy)

    fig_tree = go.Figure(go.Treemap(
        labels=labels,
        parents=parents,
        root_color="lightgrey"
    ))

    fig_tree.update_layout(
        title='Data Columns Hierarchy',
        width=850,
        height=650
    )

    tree_html = pio.to_html(fig_tree, full_html=False)

    # Generate sunburst chart
    df_counts = df.groupby(['Churn', 'Contract', 'PaymentMethod', 'InternetService']).size().reset_index(name='Count')

    fig_sunburst = px.sunburst(df_counts,
                               path=['Churn', 'Contract', 'PaymentMethod', 'InternetService'],
                               values='Count',
                               color='Churn',
                               color_continuous_scale='RdBu',
                               # title='Telco Customer Churn Sunburst Chart',
                               labels={'Churn': 'Churn', 'Contract': 'Contract', 'PaymentMethod': 'Payment Method', 'InternetService': 'Internet Service'})

    fig_sunburst.update_layout(
        transition_duration=500,
        width=850,
        height=650
    )

    sunburst_html = pio.to_html(fig_sunburst, full_html=False)

    return render_template('index.html', tree_html=tree_html, sunburst_html=sunburst_html)



if __name__ == '__main__':
    app.run(debug=True)
