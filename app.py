from flask import Flask, render_template
import pandas as pd
import plotly.express as px
import plotly.io as pio
import plotly.colors as colors

app = Flask(__name__)

@app.route('/')
def correlation_heatmap():
    # Load the dataset
    df = pd.read_csv('../WA_Fn-UseC_-Telco-Customer-Churn.csv')

    # Preprocess the data
    df = df[[i for i in df.columns if i not in ('customerID', 'Churn')]]
    df = df.drop(columns=["OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies"])
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df_n = pd.get_dummies(df, drop_first=True)
    corr_matrix = df_n.corr()

    # Create an interactive heatmap
    fig = px.imshow(corr_matrix,
                    labels=dict(color="Correlation"),
                    x=corr_matrix.columns,
                    y=corr_matrix.columns,
                    color_continuous_scale=colors.sequential.Viridis)

    # Update layout for better visualization
    fig.update_layout(
        title='Correlation Matrix Heatmap',
        xaxis_nticks=72,
        yaxis_nticks=36,
        width=1200,  # Set the width of the figure
        height=800   # Set the height of the figure
    )

    # Convert the plotly figure to HTML
    heatmap_html = pio.to_html(fig, full_html=False)

    return render_template('index.html', heatmap_html=heatmap_html)

if __name__ == '__main__':
    app.run(debug=True)