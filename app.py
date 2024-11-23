from flask import Flask, render_template, request
import pandas as pd
import plotly.express as px
import plotly.io as pio
import hiplot as hip
import tempfile

app = Flask(__name__)

# Load the dataset
df = pd.read_csv('WA_Fn-UseC_-Telco-Customer-Churn.csv')
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df_n = pd.get_dummies(df.drop(columns=['customerID', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies']), drop_first=True)
corr_matrix = df_n.corr()

@app.route('/')
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

    return render_template('index.html', heatmap_html=heatmap_html)

@app.route('/scatter_plot')
def scatter_plot():
    x = request.args.get('x')
    y = request.args.get('y')

    # Map dummy variable names back to original column names
    dummy_columns = {col: col.split('_')[0] for col in df_n.columns if '_' in col}
    x_original = dummy_columns.get(x, x)
    y_original = dummy_columns.get(y, y)

    if x_original in df.columns and y_original in df.columns:
        # Ensure the columns are numeric
        if not pd.api.types.is_numeric_dtype(df[x_original]):
            df[x_original] = pd.factorize(df[x_original])[0]
        if not pd.api.types.is_numeric_dtype(df[y_original]):
            df[y_original] = pd.factorize(df[y_original])[0]

        fig = px.scatter(df, x=x_original, y=y_original, trendline='ols')
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

if __name__ == '__main__':
    app.run(debug=True)