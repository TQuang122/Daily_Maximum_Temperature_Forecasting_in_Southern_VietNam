"""
Model Evaluation Dashboard
Interactive Plotly Dash dashboard for visualizing model evaluation results
"""

import dash
from dash import dcc, html, Input, Output, State, callback_context, dash_table
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
import json
from pathlib import Path
from datetime import datetime, date
import warnings
import base64
from scipy import stats as scipy_stats
from sklearn.metrics import mean_absolute_error

warnings.filterwarnings('ignore')

# Initialize Dash app
app = dash.Dash(__name__, suppress_callback_exceptions=True)
app.title = "Model Evaluation Dashboard"

# Color scheme for models - Modern professional palette
MODEL_COLORS = {
    'xgb': '#4A90E2',  # Modern Blue
    'rf': '#F5A623',   # Warm Orange
    'hgb': '#50C878',  # Fresh Green
    'lgb': '#E74C3C'   # Vibrant Red
}

# Extended color palette for visualizations
COLOR_PALETTE = {
    'primary': '#2C3E50',      # Dark Blue-Gray
    'secondary': '#3498DB',     # Bright Blue
    'success': '#27AE60',       # Green
    'warning': '#F39C12',       # Orange
    'danger': '#E74C3C',        # Red
    'info': '#3498DB',          # Blue
    'light': '#ECF0F1',         # Light Gray
    'dark': '#34495E',          # Dark Gray
    'accent1': '#9B59B6',       # Purple
    'accent2': '#1ABC9C',       # Turquoise
    'accent3': '#E67E22',       # Dark Orange
    'accent4': '#16A085',        # Dark Turquoise
}

# Gradient colors for charts
GRADIENT_COLORS = [
    '#667EEA',  # Purple-Blue
    '#764BA2',  # Purple
    '#F093FB',  # Pink
    '#4FACFE',  # Light Blue
    '#00F2FE',  # Cyan
    '#43E97B',  # Green
    '#FA709A',  # Rose
    '#FEE140',  # Yellow
]

# Sequential color scales
SEQUENTIAL_BLUES = ['#E3F2FD', '#90CAF9', '#42A5F5', '#1E88E5', '#1565C0']
SEQUENTIAL_ORANGES = ['#FFF3E0', '#FFB74D', '#FF9800', '#F57C00', '#E65100']
SEQUENTIAL_GREENS = ['#E8F5E9', '#81C784', '#4CAF50', '#388E3C', '#1B5E20']
SEQUENTIAL_REDS = ['#FFEBEE', '#E57373', '#EF5350', '#E53935', '#C62828']

MODEL_NAMES = {
    'xgb': 'XGBoost',
    'rf': 'Random Forest',
    'hgb': 'HistGradientBoosting',
    'lgb': 'LightGBM'
}

# Paths
PROJECT_ROOT = Path(__file__).parent
RESULTS_DIR = PROJECT_ROOT / 'results'
ARTIFACTS_DIR = PROJECT_ROOT / 'artifacts'
FIGURES_DIR = PROJECT_ROOT / 'figures'

# Data cache
data_cache = {}


def encode_image(image_path):
    """Encode image to base64 for display in Dash"""
    if not image_path.exists():
        return None
    encoded = base64.b64encode(open(image_path, 'rb').read())
    return 'data:image/png;base64,{}'.format(encoded.decode())


# ==================== DATA LOADING FUNCTIONS ====================

def load_latest_evaluation_summary():
    """Load the latest evaluation summary JSON file"""
    cache_key = 'evaluation_summary'
    if cache_key in data_cache:
        return data_cache[cache_key]
    
    summary_files = list(RESULTS_DIR.glob('evaluation_summary_*.json'))
    if not summary_files:
        return None
    
    latest_file = max(summary_files, key=lambda x: x.stat().st_mtime)
    with open(latest_file, 'r') as f:
        data = json.load(f)
    
    data_cache[cache_key] = data
    return data


def load_model_comparison():
    """Load the latest model comparison CSV"""
    cache_key = 'model_comparison'
    if cache_key in data_cache:
        return data_cache[cache_key]
    
    comparison_files = list(RESULTS_DIR.glob('model_comparison_*.csv'))
    if not comparison_files:
        return None
    
    latest_file = max(comparison_files, key=lambda x: x.stat().st_mtime)
    df = pd.read_csv(latest_file)
    
    data_cache[cache_key] = df
    return df


def load_predictions_data(model_key=None, date_start=None, date_end=None):
    """Load predictions data with optional filtering"""
    cache_key = f'predictions_{model_key}_{date_start}_{date_end}'
    if cache_key in data_cache:
        return data_cache[cache_key]
    
    if model_key:
        pred_files = list(RESULTS_DIR.glob(f'predictions_{model_key}_*.csv'))
    else:
        pred_files = list(RESULTS_DIR.glob('predictions_*.csv'))
    
    if not pred_files:
        return None
    
    all_predictions = {}
    for file in pred_files:
        model = file.stem.split('_')[1]  # Extract model key from filename
        df = pd.read_csv(file)
        df['datetime'] = pd.to_datetime(df['datetime'])
        
        # Apply date filtering
        if date_start:
            df = df[df['datetime'] >= pd.to_datetime(date_start)]
        if date_end:
            df = df[df['datetime'] <= pd.to_datetime(date_end)]
        
        all_predictions[model] = df
    
    data_cache[cache_key] = all_predictions
    return all_predictions


def get_available_dates():
    """Get available date range from predictions data"""
    pred_files = list(RESULTS_DIR.glob('predictions_*.csv'))
    if not pred_files:
        return None, None
    
    # Use first file to get date range
    df = pd.read_csv(pred_files[0])
    df['datetime'] = pd.to_datetime(df['datetime'])
    
    return df['datetime'].min().date(), df['datetime'].max().date()


def load_error_analysis():
    """Load error analysis data"""
    cache_key = 'error_analysis'
    if cache_key in data_cache:
        return data_cache[cache_key]
    
    error_data = {}
    
    # Load province errors
    province_files = list(RESULTS_DIR.glob('error_by_province_*.csv'))
    if province_files:
        latest = max(province_files, key=lambda x: x.stat().st_mtime)
        error_data['province'] = pd.read_csv(latest)
    
    # Load month errors
    month_files = list(RESULTS_DIR.glob('error_by_month_*.csv'))
    if month_files:
        latest = max(month_files, key=lambda x: x.stat().st_mtime)
        error_data['month'] = pd.read_csv(latest)
    
    # Load season errors
    season_files = list(RESULTS_DIR.glob('error_by_season_*.csv'))
    if season_files:
        latest = max(season_files, key=lambda x: x.stat().st_mtime)
        error_data['season'] = pd.read_csv(latest)
    
    data_cache[cache_key] = error_data
    return error_data


def load_feature_importance():
    """Load feature importance data"""
    cache_key = 'feature_importance'
    if cache_key in data_cache:
        return data_cache[cache_key]
    
    importance_files = list(RESULTS_DIR.glob('feature_importance_*.csv'))
    if not importance_files:
        return {}
    
    importance_data = {}
    for file in importance_files:
        model_key = file.stem.split('_')[-1]  # Extract model key
        df = pd.read_csv(file)
        importance_data[model_key] = df
    
    data_cache[cache_key] = importance_data
    return importance_data


# ==================== LAYOUT COMPONENTS ====================

def create_sidebar():
    """Create sidebar navigation"""
    return html.Div(
        [
            html.H2("Model Evaluation", className="sidebar-header"),
            html.Hr(),
            dcc.Link("Overview", href="/", className="nav-link"),
            dcc.Link("Predictions Analysis", href="/predictions", className="nav-link"),
            dcc.Link("Error Analysis - Spatial", href="/error-spatial", className="nav-link"),
            dcc.Link("Error Analysis - Temporal", href="/error-temporal", className="nav-link"),
            dcc.Link("Feature Importance", href="/feature-importance", className="nav-link"),
            dcc.Link("Advanced Analysis", href="/advanced", className="nav-link"),
            dcc.Link("Error Heatmaps", href="/heatmaps", className="nav-link"),
        ],
        className="sidebar"
    )


def create_date_picker():
    """Create date range picker component"""
    date_min, date_max = get_available_dates()
    
    return html.Div(
        [
            html.Label("Date Range Filter:", style={'fontWeight': 'bold', 'marginBottom': '5px'}),
            dcc.DatePickerRange(
                id='date-picker-range',
                start_date=date_min,
                end_date=date_max,
                min_date_allowed=date_min,
                max_date_allowed=date_max,
                display_format='YYYY-MM-DD',
                style={'width': '100%'}
            ),
        ],
        style={
            'padding': '15px', 
            'background': 'linear-gradient(135deg, #ffffff 0%, #f8f9fa 100%)', 
            'borderRadius': '10px', 
            'marginBottom': '20px',
            'boxShadow': '0 2px 4px rgba(0,0,0,0.1)',
            'border': '1px solid rgba(52, 152, 219, 0.2)'
        }
    )


# ==================== PAGE LAYOUTS ====================

def create_overview_page():
    """Create Overview page"""
    summary = load_latest_evaluation_summary()
    comparison_df = load_model_comparison()
    
    if summary is None:
        return html.Div("No evaluation data found. Please run the evaluation notebook first.")
    
    best_model = summary.get('best_model', 'rf')
    error_analysis = summary.get('error_analysis', {})
    
    # Create metrics cards
    metrics_cards = html.Div(
        [
            html.Div(
                [
                    html.H4("Best Model", style={'margin': '0'}),
                    html.H2(MODEL_NAMES.get(best_model, best_model.upper()), 
                           style={'color': MODEL_COLORS.get(best_model, '#333'), 'margin': '5px 0'}),
                ],
                className="metric-card"
            ),
            html.Div(
                [
                    html.H4("Overall MAE", style={'margin': '0'}),
                    html.H2(f"{error_analysis.get('overall_mae', 0):.3f}°C", 
                           style={'margin': '5px 0'}),
                ],
                className="metric-card"
            ),
            html.Div(
                [
                    html.H4("Overall RMSE", style={'margin': '0'}),
                    html.H2(f"{error_analysis.get('overall_rmse', 0):.3f}°C", 
                           style={'margin': '5px 0'}),
                ],
                className="metric-card"
            ),
            html.Div(
                [
                    html.H4("Provinces", style={'margin': '0'}),
                    html.H2(f"{error_analysis.get('n_provinces', 0)}", 
                           style={'margin': '5px 0'}),
                ],
                className="metric-card"
            ),
        ],
        style={'display': 'grid', 'gridTemplateColumns': 'repeat(4, 1fr)', 'gap': '20px', 'marginBottom': '30px'}
    )
    
    # Model comparison chart
    if comparison_df is not None:
        fig_comparison = go.Figure()
        
        models = comparison_df['Model'].tolist()
        mae_values = comparison_df['Test MAE (°C)'].tolist()
        rmse_values = comparison_df['Test RMSE (°C)'].tolist()
        r2_values = comparison_df['Test R²'].tolist()
        
        # Color MAE bars based on threshold (< 1°C = green, >= 1°C = red)
        mae_colors_overview = [COLOR_PALETTE['success'] if mae < 1.0 else COLOR_PALETTE['danger'] for mae in mae_values]
        
        fig_comparison.add_trace(go.Bar(
            name='MAE',
            x=models,
            y=mae_values,
            marker_color=mae_colors_overview,
            marker_line_color=COLOR_PALETTE['primary'],
            marker_line_width=1.5,
            text=[f'{mae:.3f}°C' for mae in mae_values],
            textposition='outside'
        ))
        fig_comparison.add_trace(go.Bar(
            name='RMSE',
            x=models,
            y=rmse_values,
            marker_color=COLOR_PALETTE['warning'],
            marker_line_color=COLOR_PALETTE['primary'],
            marker_line_width=1.5
        ))
        
        fig_comparison.update_layout(
            title='Model Performance Comparison - MAE: Green < 1°C, Red ≥ 1°C',
            xaxis_title='Model',
            yaxis_title='Error (°C)',
            barmode='group',
            height=400,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color=COLOR_PALETTE['primary']),
            title_font=dict(size=18, color=COLOR_PALETTE['primary']),
            shapes=[{
                'type': 'line',
                'x0': -0.5,
                'x1': len(models) - 0.5,
                'y0': 1.0,
                'y1': 1.0,
                'line': {
                    'color': COLOR_PALETTE['warning'],
                    'width': 2,
                    'dash': 'dash'
                },
                'layer': 'below'
            }]
        )
    else:
        fig_comparison = go.Figure()
    
    return html.Div(
        [
            html.H1("Overview Dashboard", style={'marginBottom': '30px'}),
            metrics_cards,
            html.Div(
                [
                    html.H3("Model Comparison Table"),
                    html.Div(
                        dash_table.DataTable(
                            data=comparison_df.to_dict('records') if comparison_df is not None else [],
                            columns=[{"name": i, "id": i} for i in comparison_df.columns] if comparison_df is not None else [],
                            style_cell={'textAlign': 'left', 'padding': '10px'},
                            style_header={'backgroundColor': COLOR_PALETTE['primary'], 'color': 'white', 'fontWeight': 'bold'},
                            style_data_conditional=[
                                {
                                    'if': {'filter_query': '{Model} = RandomForest'},
                                    'backgroundColor': '#fff3cd',
                                }
                            ],
                        ),
                        style={'marginBottom': '30px'}
                    ),
                    html.H3("Performance Charts"),
                    dcc.Graph(figure=fig_comparison),
                ]
            ),
        ],
        style={'padding': '20px'}
    )


def create_predictions_page():
    """Create Predictions Analysis page"""
    return html.Div(
        [
            html.H1("Predictions Analysis", style={'marginBottom': '30px'}),
            create_date_picker(),
            dcc.Dropdown(
                id='model-selector-predictions',
                options=[{'label': MODEL_NAMES.get(k, k), 'value': k} 
                        for k in MODEL_NAMES.keys()],
                value='rf',
                style={'width': '300px', 'marginBottom': '20px'}
            ),
            dcc.Loading(
                id="loading-predictions",
                type="default",
                children=html.Div(id='predictions-content')
            ),
        ],
        style={'padding': '20px'}
    )


def create_error_spatial_page():
    """Create Error Analysis - Spatial page (includes Performance by Province)"""
    return html.Div(
        [
            html.H1("Error Analysis - Spatial & Performance by Province", style={'marginBottom': '30px'}),
            create_date_picker(),
            html.H2("Error Analysis by Province", style={'marginTop': '30px', 'marginBottom': '20px'}),
            dcc.Loading(
                id="loading-error-spatial",
                type="default",
                children=html.Div(id='error-spatial-content')
            ),
            html.H2("Model Performance by Province", style={'marginTop': '40px', 'marginBottom': '20px'}),
            dcc.Loading(
                id="loading-province-performance",
                type="default",
                children=html.Div(id='province-performance-content')
            ),
        ],
        style={'padding': '20px'}
    )


def create_error_temporal_page():
    """Create Error Analysis - Temporal page"""
    return html.Div(
        [
            html.H1("Error Analysis - Temporal", style={'marginBottom': '30px'}),
            create_date_picker(),
            dcc.Loading(
                id="loading-error-temporal",
                type="default",
                children=html.Div(id='error-temporal-content')
            ),
        ],
        style={'padding': '20px'}
    )


def create_feature_importance_page():
    """Create Feature Importance page"""
    return html.Div(
        [
            html.H1("Feature Importance Analysis", style={'marginBottom': '30px'}),
            html.Div([
                html.Label("Select Model:", style={'fontWeight': 'bold', 'marginRight': '10px'}),
                dcc.Dropdown(
                    id='model-selector-importance',
                    options=[{'label': MODEL_NAMES.get(k, k), 'value': k}
                            for k in MODEL_NAMES.keys()],
                    value='rf',
                    style={'width': '300px', 'display': 'inline-block'}
                ),
                html.Label("Top N Features:", style={'fontWeight': 'bold', 'marginLeft': '30px', 'marginRight': '10px'}),
                dcc.Slider(
                    id='top-n-slider',
                    min=5,
                    max=20,
                    step=1,
                    value=15,
                    marks={i: str(i) for i in range(5, 21, 5)},
                    tooltip={"placement": "bottom", "always_visible": True}
                ),
            ], style={'marginBottom': '30px'}),
            dcc.Loading(
                id="loading-feature-importance",
                type="default",
                children=html.Div(id='feature-importance-content')
            ),
        ],
        style={'padding': '20px'}
    )


def create_advanced_analysis_page():
    """Create Advanced Analysis page"""
    return html.Div(
        [
            html.H1("Advanced Analysis", style={'marginBottom': '30px'}),
            create_date_picker(),
            html.Div([
                html.Label("Select Model:", style={'fontWeight': 'bold', 'marginRight': '10px'}),
                dcc.Dropdown(
                    id='model-selector-advanced',
                    options=[{'label': MODEL_NAMES.get(k, k), 'value': k}
                            for k in MODEL_NAMES.keys()],
                    value='rf',
                    style={'width': '300px'}
                ),
            ], style={'marginBottom': '30px'}),
            html.H2("Residual Analysis", style={'marginTop': '30px', 'marginBottom': '20px'}),
            dcc.Loading(
                id="loading-residual-analysis",
                type="default",
                children=html.Div(id='residual-analysis-content')
            ),
            html.H2("Temporal Analysis", style={'marginTop': '40px', 'marginBottom': '20px'}),
            dcc.Loading(
                id="loading-temporal-analysis",
                type="default",
                children=html.Div(id='temporal-analysis-content')
            ),
        ],
        style={'padding': '20px'}
    )


def create_heatmaps_page():
    """Create Error Heatmaps page"""
    return html.Div(
        [
            html.H1("Error Heatmaps - Spatiotemporal Analysis", style={'marginBottom': '30px'}),
            create_date_picker(),
            html.Div([
                html.Label("Select Model:", style={'fontWeight': 'bold', 'marginRight': '10px'}),
                dcc.Dropdown(
                    id='model-selector-heatmap',
                    options=[{'label': MODEL_NAMES.get(k, k), 'value': k}
                            for k in MODEL_NAMES.keys()],
                    value='rf',
                    style={'width': '300px'}
                ),
            ], style={'marginBottom': '30px'}),
            dcc.Loading(
                id="loading-heatmaps",
                type="default",
                children=html.Div(id='heatmaps-content')
            ),
        ],
        style={'padding': '20px'}
    )


# ==================== MAIN APP LAYOUT ====================

app.layout = html.Div(
    [
        dcc.Location(id='url', refresh=False),
        create_sidebar(),
        html.Div(id='page-content', className="content"),
    ],
    style={'display': 'flex'}
)


# ==================== CALLBACKS ====================

@app.callback(Output('page-content', 'children'),
              Input('url', 'pathname'))
def display_page(pathname):
    """Route to different pages based on URL"""
    if pathname == '/predictions':
        return create_predictions_page()
    elif pathname == '/error-spatial':
        return create_error_spatial_page()
    elif pathname == '/error-temporal':
        return create_error_temporal_page()
    elif pathname == '/feature-importance':
        return create_feature_importance_page()
    elif pathname == '/advanced':
        return create_advanced_analysis_page()
    elif pathname == '/heatmaps':
        return create_heatmaps_page()
    else:
        return create_overview_page()


# Predictions page callback
@app.callback(
    Output('predictions-content', 'children'),
    [Input('model-selector-predictions', 'value'),
     Input('date-picker-range', 'start_date'),
     Input('date-picker-range', 'end_date')]
)
def update_predictions(model_key, start_date, end_date):
    """Update predictions visualizations"""
    try:
        predictions = load_predictions_data(model_key=model_key, date_start=start_date, date_end=end_date)
        
        if not predictions or model_key not in predictions:
            return html.Div(
                "No predictions data available for selected model.",
                className="error-message"
            )
        
        df = predictions[model_key]
        
        # Scatter plot: Predictions vs Actual
        fig_scatter = px.scatter(
            df,
            x='actual',
            y='predicted',
            title=f'Predictions vs Actual - {MODEL_NAMES.get(model_key, model_key)}',
            labels={'actual': 'Actual Temperature (°C)', 'predicted': 'Predicted Temperature (°C)'},
            color_discrete_sequence=[MODEL_COLORS.get(model_key, COLOR_PALETTE['secondary'])]
        )
        
        # Add perfect prediction line
        min_val = min(df['actual'].min(), df['predicted'].min())
        max_val = max(df['actual'].max(), df['predicted'].max())
        fig_scatter.add_trace(go.Scatter(
            x=[min_val, max_val],
            y=[min_val, max_val],
            mode='lines',
            name='Perfect Prediction',
            line=dict(color=COLOR_PALETTE['danger'], dash='dash', width=2)
        ))
        
        # Residual plot
        df['residual'] = df['actual'] - df['predicted']
        fig_residual = px.scatter(
            df,
            x='predicted',
            y='residual',
            title='Residuals vs Predicted',
            labels={'predicted': 'Predicted Temperature (°C)', 'residual': 'Residual (°C)'},
            color_discrete_sequence=[COLOR_PALETTE['accent2']]
        )
        fig_residual.add_hline(y=0, line_dash="dash", line_color=COLOR_PALETTE['danger'], line_width=2)
        
        # Temporal line chart
        df_sorted = df.sort_values('datetime')
        fig_temporal = go.Figure()
        fig_temporal.add_trace(go.Scatter(
            x=df_sorted['datetime'],
            y=df_sorted['actual'],
            name='Actual',
            line=dict(color=COLOR_PALETTE['secondary'], width=2.5)
        ))
        fig_temporal.add_trace(go.Scatter(
            x=df_sorted['datetime'],
            y=df_sorted['predicted'],
            name='Predicted',
            line=dict(color=COLOR_PALETTE['accent1'], dash='dash', width=2.5)
        ))
        fig_temporal.update_layout(
            title='Temporal Predictions',
            xaxis_title='Date',
            yaxis_title='Temperature (°C)',
            height=400,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color=COLOR_PALETTE['primary']),
            title_font=dict(size=18, color=COLOR_PALETTE['primary'])
        )
        
        return html.Div([
            dcc.Graph(figure=fig_scatter),
            dcc.Graph(figure=fig_residual),
            dcc.Graph(figure=fig_temporal),
        ])
    except Exception as e:
        return html.Div(
            f"Error loading predictions data: {str(e)}",
            className="error-message"
        )


# Error Spatial page callback
@app.callback(
    Output('error-spatial-content', 'children'),
    [Input('date-picker-range', 'start_date'),
     Input('date-picker-range', 'end_date')]
)
def update_error_spatial(start_date, end_date):
    """Update spatial error analysis"""
    try:
        error_data = load_error_analysis()
        
        if 'province' not in error_data:
            return html.Div(
                "No spatial error data available. Please run the evaluation notebook first.",
                className="error-message"
            )
        
        df = error_data['province']
        
        # Province error bar chart with conditional coloring (green < 1°C, red >= 1°C)
        fig_bar = go.Figure()
        
        # Create color array based on threshold
        colors = [COLOR_PALETTE['success'] if mae < 1.0 else COLOR_PALETTE['danger'] for mae in df['MAE']]
        
        fig_bar.add_trace(go.Bar(
            x=df['name'],
            y=df['MAE'],
            error_y=dict(type='data', array=df['MAE_std'], color=COLOR_PALETTE['dark']),
            marker_color=colors,
            marker_line_color=COLOR_PALETTE['primary'],
            marker_line_width=1.5,
            name='MAE',
            text=[f'{mae:.2f}°C' for mae in df['MAE']],
            textposition='outside'
        ))
        fig_bar.update_layout(
            title='MAE by Province (Green: < 1°C, Red: ≥ 1°C)',
            xaxis_title='Province',
            yaxis_title='MAE (°C)',
            height=500,
            xaxis_tickangle=-45,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color=COLOR_PALETTE['primary']),
            title_font=dict(size=18, color=COLOR_PALETTE['primary']),
            shapes=[{
                'type': 'line',
                'x0': -0.5,
                'x1': len(df) - 0.5,
                'y0': 1.0,
                'y1': 1.0,
                'line': {
                    'color': COLOR_PALETTE['warning'],
                    'width': 2,
                    'dash': 'dash'
                }
            }],
            annotations=[{
                'x': len(df) / 2,
                'y': 1.0,
                'text': 'Target: 1°C',
                'showarrow': False,
                'xref': 'x',
                'yref': 'y',
                'bgcolor': 'rgba(255,255,255,0.8)',
                'bordercolor': COLOR_PALETTE['warning'],
                'borderwidth': 1
            }]
        )
        
        return html.Div([
            dcc.Graph(figure=fig_bar),
        ])
    except Exception as e:
        return html.Div(
            f"Error loading spatial error data: {str(e)}",
            className="error-message"
        )


# Error Temporal page callback
@app.callback(
    Output('error-temporal-content', 'children'),
    [Input('date-picker-range', 'start_date'),
     Input('date-picker-range', 'end_date')]
)
def update_error_temporal(start_date, end_date):
    """Update temporal error analysis"""
    try:
        error_data = load_error_analysis()
        
        if 'month' not in error_data:
            return html.Div(
                "No temporal error data available. Please run the evaluation notebook first.",
                className="error-message"
            )
        
        month_df = error_data['month']
        
        # Error by month
        month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                      'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        month_df['MonthName'] = month_df['Month'].apply(lambda x: month_names[x-1] if 1 <= x <= 12 else f'Month {x}')
        
        fig_month = go.Figure()
        
        # Color based on threshold: green < 1°C, red >= 1°C
        month_colors = [COLOR_PALETTE['success'] if mae < 1.0 else COLOR_PALETTE['danger'] for mae in month_df['MAE']]
        
        fig_month.add_trace(go.Bar(
            x=month_df['MonthName'],
            y=month_df['MAE'],
            error_y=dict(type='data', array=month_df['MAE_std'], color=COLOR_PALETTE['dark']),
            marker_color=month_colors,
            marker_line_color=COLOR_PALETTE['primary'],
            marker_line_width=1.5,
            name='MAE',
            text=[f'{mae:.2f}°C' for mae in month_df['MAE']],
            textposition='outside'
        ))
        fig_month.update_layout(
            title='MAE by Month (Green: < 1°C, Red: ≥ 1°C)',
            xaxis_title='Month',
            yaxis_title='MAE (°C)',
            height=400,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color=COLOR_PALETTE['primary']),
            title_font=dict(size=18, color=COLOR_PALETTE['primary']),
            shapes=[{
                'type': 'line',
                'x0': -0.5,
                'x1': len(month_df) - 0.5,
                'y0': 1.0,
                'y1': 1.0,
                'line': {
                    'color': COLOR_PALETTE['warning'],
                    'width': 2,
                    'dash': 'dash'
                },
                'layer': 'below'
            }]
        )
        
        # Error by season
        if 'season' in error_data:
            season_df = error_data['season']
            fig_season = go.Figure()
            
            # Define colors for Dry and Wet seasons
            season_color_map = {
                'Dry': COLOR_PALETTE['accent3'],  # Dark Orange for Dry
                'Wet': COLOR_PALETTE['accent2']   # Turquoise for Wet
            }
            
            # Create color array based on season type
            season_colors = [season_color_map.get(season, COLOR_PALETTE['secondary']) for season in season_df['season']]
            
            fig_season.add_trace(go.Bar(
                x=season_df['season'],
                y=season_df['MAE'],
                error_y=dict(type='data', array=season_df['MAE_std'], color=COLOR_PALETTE['dark']),
                marker_color=season_colors,
                marker_line_color=COLOR_PALETTE['primary'],
                marker_line_width=1.5,
                name='MAE',
                text=[f'{mae:.2f}°C' for mae in season_df['MAE']],
                textposition='outside'
            ))
            fig_season.update_layout(
                title='MAE by Season (Dry: Orange, Wet: Turquoise)',
                xaxis_title='Season',
                yaxis_title='MAE (°C)',
                height=400,
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color=COLOR_PALETTE['primary']),
                title_font=dict(size=18, color=COLOR_PALETTE['primary'])
            )
        else:
            fig_season = go.Figure()
        
        return html.Div([
            dcc.Graph(figure=fig_month),
            dcc.Graph(figure=fig_season),
        ])
    except Exception as e:
        return html.Div(
            f"Error loading temporal error data: {str(e)}",
            className="error-message"
        )


# Province Performance page callback
@app.callback(
    Output('province-performance-content', 'children'),
    [Input('date-picker-range', 'start_date'),
     Input('date-picker-range', 'end_date')]
)
def update_province_performance(start_date, end_date):
    """Update province performance visualization"""
    try:
        predictions = load_predictions_data(date_start=start_date, date_end=end_date)
        
        if not predictions:
            return html.Div(
                "No predictions data available. Please run the evaluation notebook first.",
                className="error-message"
            )
        
        # Calculate MAE by province for each model
        province_performance = []
        for model_key, df in predictions.items():
            province_stats = df.groupby('name')['abs_error'].agg(['mean', 'std']).reset_index()
            province_stats.columns = ['province', 'mae', 'std']
            province_stats['model'] = MODEL_NAMES.get(model_key, model_key)
            province_performance.append(province_stats)
        
        if not province_performance:
            return html.Div("No data available.", className="error-message")
        
        combined_df = pd.concat(province_performance, ignore_index=True)
        provinces = sorted(combined_df['province'].unique())
        
        # Create grouped bar chart with conditional coloring
        fig = go.Figure()
        
        for model in combined_df['model'].unique():
            model_data = combined_df[combined_df['model'] == model]
            mae_values = []
            std_values = []
            
            for province in provinces:
                prov_data = model_data[model_data['province'] == province]
                if len(prov_data) > 0:
                    mae_values.append(prov_data['mae'].iloc[0])
                    std_values.append(prov_data['std'].iloc[0] if not pd.isna(prov_data['std'].iloc[0]) else 0)
                else:
                    mae_values.append(0)
                    std_values.append(0)
            
            # Use original model colors (no conditional coloring)
            color = MODEL_COLORS.get([k for k, v in MODEL_NAMES.items() if v == model][0], '#808080')
            
            fig.add_trace(go.Bar(
                name=model,
                x=provinces,
                y=mae_values,
                error_y=dict(type='data', array=std_values, color=COLOR_PALETTE['dark']),
                marker_color=color,
                marker_line_color=COLOR_PALETTE['primary'],
                marker_line_width=1.5
            ))
        
        fig.update_layout(
            title='Model Performance by Province',
            xaxis_title='Province',
            yaxis_title='MAE (°C)',
            barmode='group',
            height=600,
            xaxis_tickangle=-45,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color=COLOR_PALETTE['primary']),
            title_font=dict(size=18, color=COLOR_PALETTE['primary'])
        )
        
        return html.Div([
            dcc.Graph(figure=fig),
        ])
    except Exception as e:
        return html.Div(
            f"Error loading province performance data: {str(e)}",
            className="error-message"
        )


# Feature Importance page callback
@app.callback(
    Output('feature-importance-content', 'children'),
    [Input('model-selector-importance', 'value'),
     Input('top-n-slider', 'value')]
)
def update_feature_importance(model_key, top_n):
    """Update feature importance visualizations"""
    try:
        importance_data = load_feature_importance()

        if not importance_data:
            # Try to show fallback image
            fig_path = FIGURES_DIR / 'feature_importance_all_models.png'
            if fig_path.exists():
                return html.Div([
                    html.P("Feature importance CSV files not found. Showing static image:",
                           className="info-message"),
                    html.Img(src=encode_image(fig_path), style={'width': '100%', 'maxWidth': '1200px'})
                ])
            return html.Div(
                "No feature importance data available. Please run the evaluation notebook first.",
                className="error-message"
            )

        if model_key not in importance_data:
            return html.Div(
                f"No feature importance data available for {MODEL_NAMES.get(model_key, model_key)}.",
                className="error-message"
            )

        df = importance_data[model_key]
        df_top = df.head(top_n)

        # Create horizontal bar chart
        fig = go.Figure()
        fig.add_trace(go.Bar(
            y=df_top['feature'][::-1],  # Reverse for better visualization
            x=df_top['importance'][::-1],
            orientation='h',
            marker_color=MODEL_COLORS.get(model_key, COLOR_PALETTE['secondary']),
            marker_line_color=COLOR_PALETTE['primary'],
            marker_line_width=1.5,
            text=[f'{imp:.4f}' for imp in df_top['importance'][::-1]],
            textposition='outside'
        ))

        fig.update_layout(
            title=f'Feature Importance - {MODEL_NAMES.get(model_key, model_key)} (Top {top_n})',
            xaxis_title='Importance (Normalized)',
            yaxis_title='Feature',
            height=max(400, top_n * 25),
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color=COLOR_PALETTE['primary']),
            title_font=dict(size=18, color=COLOR_PALETTE['primary'])
        )

        return html.Div([
            dcc.Graph(figure=fig),
            html.Div([
                html.H3("Top 10 Features", style={'marginTop': '30px'}),
                dash_table.DataTable(
                    data=df.head(10).to_dict('records'),
                    columns=[{"name": i.title(), "id": i} for i in df.columns],
                    style_cell={'textAlign': 'left', 'padding': '10px'},
                    style_header={'backgroundColor': COLOR_PALETTE['primary'], 'color': 'white', 'fontWeight': 'bold'},
                    style_data={'backgroundColor': 'white'},
                )
            ])
        ])
    except Exception as e:
        return html.Div(
            f"Error loading feature importance data: {str(e)}",
            className="error-message"
        )


# Advanced Analysis - Residual Analysis callback
@app.callback(
    Output('residual-analysis-content', 'children'),
    [Input('model-selector-advanced', 'value'),
     Input('date-picker-range', 'start_date'),
     Input('date-picker-range', 'end_date')]
)
def update_residual_analysis(model_key, start_date, end_date):
    """Update residual analysis visualizations"""
    try:
        predictions = load_predictions_data(model_key=model_key, date_start=start_date, date_end=end_date)

        if not predictions or model_key not in predictions:
            return html.Div(
                "No predictions data available for selected model.",
                className="error-message"
            )

        df = predictions[model_key]
        df['residual'] = df['actual'] - df['predicted']
        residuals = df['residual']

        # Create 2x2 subplot figure
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                'Residual Distribution',
                'Residuals vs Predicted',
                'Q-Q Plot',
                'Residuals by Model Comparison'
            ),
            specs=[[{}, {}], [{'type': 'scatter'}, {'type': 'box'}]]
        )

        # 1. Residual Distribution (histogram)
        fig.add_trace(
            go.Histogram(x=residuals, nbinsx=50, marker_color=COLOR_PALETTE['secondary'],
                        marker_line_color=COLOR_PALETTE['primary'], marker_line_width=1,
                        name='Distribution'),
            row=1, col=1
        )
        fig.add_vline(x=0, line_dash="dash", line_color=COLOR_PALETTE['danger'],
                      line_width=2, row=1, col=1)

        # 2. Residuals vs Predicted
        fig.add_trace(
            go.Scatter(x=df['predicted'], y=residuals, mode='markers',
                      marker=dict(color=COLOR_PALETTE['accent2'], size=5, opacity=0.5),
                      name='Residuals'),
            row=1, col=2
        )
        fig.add_hline(y=0, line_dash="dash", line_color=COLOR_PALETTE['danger'],
                     line_width=2, row=1, col=2)

        # 3. Q-Q Plot
        (osm, osr), (slope, intercept, r) = scipy_stats.probplot(residuals, dist="norm")
        fig.add_trace(
            go.Scatter(x=osm, y=osr, mode='markers',
                      marker=dict(color=COLOR_PALETTE['secondary'], size=5),
                      name='Q-Q'),
            row=2, col=1
        )
        fig.add_trace(
            go.Scatter(x=osm, y=slope * osm + intercept, mode='lines',
                      line=dict(color=COLOR_PALETTE['danger'], dash='dash'),
                      name='Reference Line'),
            row=2, col=1
        )

        # 4. Box plot of residuals
        fig.add_trace(
            go.Box(y=residuals, name=MODEL_NAMES.get(model_key, model_key),
                  marker_color=MODEL_COLORS.get(model_key, COLOR_PALETTE['secondary']),
                  boxmean='sd'),
            row=2, col=2
        )

        fig.update_xaxes(title_text="Residuals (°C)", row=1, col=1)
        fig.update_yaxes(title_text="Frequency", row=1, col=1)
        fig.update_xaxes(title_text="Predicted (°C)", row=1, col=2)
        fig.update_yaxes(title_text="Residuals (°C)", row=1, col=2)
        fig.update_xaxes(title_text="Theoretical Quantiles", row=2, col=1)
        fig.update_yaxes(title_text="Sample Quantiles", row=2, col=1)
        fig.update_yaxes(title_text="Residuals (°C)", row=2, col=2)

        fig.update_layout(
            height=800,
            showlegend=False,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color=COLOR_PALETTE['primary']),
            title_font=dict(size=16, color=COLOR_PALETTE['primary'])
        )

        # Statistics
        mean_res = residuals.mean()
        std_res = residuals.std()

        stats_div = html.Div([
            html.H3("Residual Statistics", style={'marginTop': '30px'}),
            html.Div([
                html.P(f"Mean: {mean_res:.4f}°C", style={'margin': '5px 0'}),
                html.P(f"Std Dev: {std_res:.4f}°C", style={'margin': '5px 0'}),
                html.P(f"Min: {residuals.min():.4f}°C", style={'margin': '5px 0'}),
                html.P(f"Max: {residuals.max():.4f}°C", style={'margin': '5px 0'}),
            ], style={'padding': '15px', 'background': COLOR_PALETTE['light'],
                     'borderRadius': '8px', 'marginTop': '10px'})
        ])

        return html.Div([
            dcc.Graph(figure=fig),
            stats_div
        ])
    except Exception as e:
        return html.Div(
            f"Error loading residual analysis data: {str(e)}",
            className="error-message"
        )


# Advanced Analysis - Temporal Analysis callback
@app.callback(
    Output('temporal-analysis-content', 'children'),
    [Input('model-selector-advanced', 'value'),
     Input('date-picker-range', 'start_date'),
     Input('date-picker-range', 'end_date')]
)
def update_temporal_analysis(model_key, start_date, end_date):
    """Update temporal analysis visualizations (Rolling MAE, Error Over Time, All Models Comparison)"""
    try:
        predictions = load_predictions_data(date_start=start_date, date_end=end_date)

        if not predictions:
            return html.Div(
                "No predictions data available.",
                className="error-message"
            )

        # Rolling MAE chart for selected model
        if model_key in predictions:
            df = predictions[model_key].copy()
            df = df.sort_values('datetime').reset_index(drop=True)

            # Calculate rolling MAE
            window = 30
            rolling_mae = []
            for i in range(len(df)):
                start_idx = max(0, i - window + 1)
                end_idx = i + 1
                window_actual = df.loc[start_idx:end_idx, 'actual']
                window_pred = df.loc[start_idx:end_idx, 'predicted']
                if len(window_actual) > 0:
                    mae_window = mean_absolute_error(window_actual, window_pred)
                    rolling_mae.append(mae_window)
                else:
                    rolling_mae.append(np.nan)

            df['rolling_mae'] = rolling_mae

            overall_mae = mean_absolute_error(df['actual'], df['predicted'])

            fig_rolling = go.Figure()
            fig_rolling.add_trace(go.Scatter(
                x=df['datetime'],
                y=df['rolling_mae'],
                mode='lines',
                name=f'{window}-Day Rolling MAE',
                line=dict(color=COLOR_PALETTE['danger'], width=2)
            ))
            fig_rolling.add_hline(
                y=overall_mae,
                line_dash="dash",
                line_color=COLOR_PALETTE['primary'],
                line_width=2,
                annotation_text=f'Overall MAE: {overall_mae:.3f}°C',
                annotation_position="top right"
            )
            fig_rolling.update_layout(
                title=f'Rolling MAE Over Time - {MODEL_NAMES.get(model_key, model_key)} ({window}-Day Window)',
                xaxis_title='Date',
                yaxis_title='MAE (°C)',
                height=400,
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color=COLOR_PALETTE['primary']),
                title_font=dict(size=18, color=COLOR_PALETTE['primary'])
            )

            # Error over time chart
            df['error'] = df['actual'] - df['predicted']
            df['abs_error'] = df['error'].abs()

            fig_error = go.Figure()
            fig_error.add_trace(go.Scatter(
                x=df['datetime'],
                y=df['error'],
                mode='lines',
                name='Error',
                line=dict(color=COLOR_PALETTE['warning'], width=1.5),
                fill='tozeroy',
                fillcolor=f'rgba({int(COLOR_PALETTE["warning"][1:3], 16)}, {int(COLOR_PALETTE["warning"][3:5], 16)}, {int(COLOR_PALETTE["warning"][5:7], 16)}, 0.2)'
            ))
            fig_error.add_hline(y=0, line_dash="dash", line_color=COLOR_PALETTE['primary'], line_width=1)
            fig_error.update_layout(
                title='Prediction Error Over Time',
                xaxis_title='Date',
                yaxis_title='Error (Actual - Predicted) (°C)',
                height=400,
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color=COLOR_PALETTE['primary']),
                title_font=dict(size=18, color=COLOR_PALETTE['primary'])
            )
        else:
            fig_rolling = go.Figure()
            fig_error = go.Figure()

        # All models comparison over time (sample period)
        fig_all_models = go.Figure()

        # Get a representative sample period (90 days from middle)
        if model_key in predictions:
            df_ref = predictions[model_key].copy()
            df_ref = df_ref.sort_values('datetime')
            start_date_data = df_ref['datetime'].min()
            end_date_data = df_ref['datetime'].max()
            mid_date = start_date_data + (end_date_data - start_date_data) / 2
            period_start = mid_date - pd.Timedelta(days=45)
            period_end = mid_date + pd.Timedelta(days=45)

            # Plot actual
            period_df = df_ref[(df_ref['datetime'] >= period_start) & (df_ref['datetime'] <= period_end)]
            fig_all_models.add_trace(go.Scatter(
                x=period_df['datetime'],
                y=period_df['actual'],
                mode='lines',
                name='Actual',
                line=dict(color=COLOR_PALETTE['primary'], width=2.5)
            ))

            # Plot all models
            model_colors = {
                'xgb': '#1f77b4',
                'lgb': '#d62728',
                'hgb': '#2ca02c',
                'rf': '#ff7f0e'
            }

            for mk in predictions.keys():
                df_model = predictions[mk].copy()
                df_model = df_model.sort_values('datetime')
                period_df_model = df_model[(df_model['datetime'] >= period_start) & (df_model['datetime'] <= period_end)]

                fig_all_models.add_trace(go.Scatter(
                    x=period_df_model['datetime'],
                    y=period_df_model['predicted'],
                    mode='lines',
                    name=MODEL_NAMES.get(mk, mk),
                    line=dict(color=model_colors.get(mk, '#808080'), width=2, dash='dash')
                ))

            fig_all_models.update_layout(
                title='All Models Comparison Over Time (Sample Period: 90 Days)',
                xaxis_title='Date',
                yaxis_title='Temperature (°C)',
                height=500,
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color=COLOR_PALETTE['primary']),
                title_font=dict(size=18, color=COLOR_PALETTE['primary'])
            )

        return html.Div([
            dcc.Graph(figure=fig_rolling),
            dcc.Graph(figure=fig_error),
            dcc.Graph(figure=fig_all_models),
        ])
    except Exception as e:
        return html.Div(
            f"Error loading temporal analysis data: {str(e)}",
            className="error-message"
        )


# Error Heatmaps page callback
@app.callback(
    Output('heatmaps-content', 'children'),
    [Input('model-selector-heatmap', 'value'),
     Input('date-picker-range', 'start_date'),
     Input('date-picker-range', 'end_date')]
)
def update_heatmaps(model_key, start_date, end_date):
    """Update error heatmap visualizations"""
    try:
        predictions = load_predictions_data(model_key=model_key, date_start=start_date, date_end=end_date)

        if not predictions or model_key not in predictions:
            return html.Div(
                "No predictions data available for selected model.",
                className="error-message"
            )

        df = predictions[model_key].copy()

        # Add season information
        df['month'] = df['datetime'].dt.month
        df['season'] = df['month'].map(lambda m: 'Dry' if m in [11, 12, 1, 2, 3, 4] else 'Wet')

        # Calculate MAE by province and season
        heatmap_data = df.groupby(['name', 'season'])['abs_error'].mean().reset_index()
        heatmap_pivot = heatmap_data.pivot(index='name', columns='season', values='abs_error')

        # Create heatmap
        fig_heatmap = go.Figure(data=go.Heatmap(
            z=heatmap_pivot.values,
            x=heatmap_pivot.columns,
            y=heatmap_pivot.index,
            colorscale='RdYlGn_r',
            text=np.round(heatmap_pivot.values, 3),
            texttemplate='%{text}°C',
            textfont={"size": 10},
            colorbar=dict(title="MAE (°C)")
        ))

        fig_heatmap.update_layout(
            title=f'Error Heatmap: Province × Season - {MODEL_NAMES.get(model_key, model_key)}',
            xaxis_title='Season',
            yaxis_title='Province',
            height=600,
            font=dict(color=COLOR_PALETTE['primary']),
            title_font=dict(size=18, color=COLOR_PALETTE['primary'])
        )

        # Calculate bias by province and season
        bias_data = df.copy()
        bias_data['error'] = bias_data['actual'] - bias_data['predicted']
        bias_heatmap_data = bias_data.groupby(['name', 'season'])['error'].mean().reset_index()
        bias_heatmap_pivot = bias_heatmap_data.pivot(index='name', columns='season', values='error')

        # Create bias heatmap
        fig_bias_heatmap = go.Figure(data=go.Heatmap(
            z=bias_heatmap_pivot.values,
            x=bias_heatmap_pivot.columns,
            y=bias_heatmap_pivot.index,
            colorscale='RdBu',
            zmid=0,
            text=np.round(bias_heatmap_pivot.values, 3),
            texttemplate='%{text}°C',
            textfont={"size": 10},
            colorbar=dict(title="Bias (°C)")
        ))

        fig_bias_heatmap.update_layout(
            title=f'Bias Heatmap: Province × Season - {MODEL_NAMES.get(model_key, model_key)}',
            xaxis_title='Season',
            yaxis_title='Province',
            height=600,
            font=dict(color=COLOR_PALETTE['primary']),
            title_font=dict(size=18, color=COLOR_PALETTE['primary'])
        )

        return html.Div([
            html.P("MAE Heatmap shows prediction error magnitude. Lower values (green) indicate better performance.",
                   style={'marginBottom': '20px'}),
            dcc.Graph(figure=fig_heatmap),
            html.P("Bias Heatmap shows prediction direction. Positive (red) = overprediction, Negative (blue) = underprediction.",
                   style={'marginTop': '40px', 'marginBottom': '20px'}),
            dcc.Graph(figure=fig_bias_heatmap),
        ])
    except Exception as e:
        return html.Div(
            f"Error loading heatmap data: {str(e)}",
            className="error-message"
        )


# ==================== CSS STYLING ====================

app.index_string = '''
<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>{%title%}</title>
        {%favicon%}
        {%css%}
        <style>
            body {
                margin: 0;
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, sans-serif;
            }
            .sidebar {
                width: 250px;
                height: 100vh;
                background: linear-gradient(180deg, #2C3E50 0%, #34495E 100%);
                padding: 20px;
                position: fixed;
                overflow-y: auto;
                box-shadow: 2px 0 10px rgba(0,0,0,0.1);
            }
            .sidebar-header {
                color: white;
                margin-bottom: 20px;
                font-weight: 600;
                text-shadow: 1px 1px 2px rgba(0,0,0,0.2);
            }
            .nav-link {
                display: block;
                color: #ECF0F1;
                padding: 12px 15px;
                margin: 5px 0;
                text-decoration: none;
                border-radius: 8px;
                transition: all 0.3s ease;
                font-weight: 500;
            }
            .nav-link:hover {
                background: linear-gradient(90deg, #3498DB 0%, #2980B9 100%);
                color: white;
                transform: translateX(5px);
                box-shadow: 0 2px 8px rgba(52, 152, 219, 0.3);
            }
            .content {
                margin-left: 250px;
                padding: 20px;
                width: calc(100% - 250px);
            }
            .metric-card {
                background: linear-gradient(135deg, #ffffff 0%, #f8f9fa 100%);
                padding: 20px;
                border-radius: 12px;
                box-shadow: 0 4px 6px rgba(0,0,0,0.1), 0 2px 4px rgba(0,0,0,0.06);
                text-align: center;
                border: 1px solid rgba(52, 152, 219, 0.1);
                transition: transform 0.2s, box-shadow 0.2s;
            }
            .metric-card:hover {
                transform: translateY(-2px);
                box-shadow: 0 8px 12px rgba(0,0,0,0.15), 0 4px 6px rgba(0,0,0,0.1);
            }
            .error-message {
                background: linear-gradient(135deg, #FFE5E5 0%, #FFCCCC 100%);
                color: #C0392B;
                padding: 15px 20px;
                border-radius: 8px;
                margin: 20px 0;
                border-left: 4px solid #E74C3C;
                box-shadow: 0 2px 4px rgba(231, 76, 60, 0.2);
            }
            .info-message {
                background: linear-gradient(135deg, #E8F4F8 0%, #D1ECF1 100%);
                color: #0C5460;
                padding: 15px 20px;
                border-radius: 8px;
                margin: 20px 0;
                border-left: 4px solid #3498DB;
                box-shadow: 0 2px 4px rgba(52, 152, 219, 0.2);
            }
        </style>
    </head>
    <body>
        {%app_entry%}
        <footer>
            {%config%}
            {%scripts%}
            {%renderer%}
        </footer>
    </body>
</html>
'''


if __name__ == '__main__':
    # For local network access, use host='0.0.0.0'
    # Others on same network can access via: http://YOUR_IP:8051
    app.run(debug=True, host='0.0.0.0', port=8051)

