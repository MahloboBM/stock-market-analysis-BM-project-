# Extracted from Plotting_data_&draft_model.ipynb

# ===== CELL 3 =====
# CELL 6: Cross-Sectional Analysis Between Target Variables
# ========================================================

import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from scipy import stats

print("🔄 CROSS-SECTIONAL ANALYSIS: TARGET VARIABLE RELATIONSHIPS")
print("=" * 65)

# Define our target variables
target_vars = ['forward_returns', 'risk_free_rate', 'market_forward_excess_returns']
target_labels = ['Forward Returns', 'Risk-Free Rate', 'Market Excess Returns']

# Create correlation matrix between target variables
target_data = train_df[target_vars].dropna()
target_corr_matrix = target_data.corr()

print(f"Analysis based on {len(target_data):,} complete observations\n")

# Print correlation matrix
print("📊 TARGET VARIABLE CORRELATION MATRIX:")
print("-" * 50)
print(f"{'':20s} {'Forward Ret':<12s} {'Risk-Free':<12s} {'Excess Ret':<12s}")
print("-" * 50)

for i, target in enumerate(target_vars):
    row_label = target_labels[i]
    values = []
    for j, other_target in enumerate(target_vars):
        corr_val = target_corr_matrix.loc[target, other_target]
        if i == j:
            values.append("1.0000")
        else:
            values.append(f"{corr_val:+.4f}")
    
    print(f"{row_label:<20s} {values[0]:<12s} {values[1]:<12s} {values[2]:<12s}")

# Calculate and display key relationships
print(f"\n🔍 KEY RELATIONSHIPS:")
print("-" * 30)

# Forward Returns vs Risk-Free Rate
fr_rf_corr = target_corr_matrix.loc['forward_returns', 'risk_free_rate']
print(f"Forward Returns ↔ Risk-Free Rate:     {fr_rf_corr:+.4f}")

# Forward Returns vs Excess Returns  
fr_ex_corr = target_corr_matrix.loc['forward_returns', 'market_forward_excess_returns']
print(f"Forward Returns ↔ Excess Returns:     {fr_ex_corr:+.4f}")

# Risk-Free Rate vs Excess Returns
rf_ex_corr = target_corr_matrix.loc['risk_free_rate', 'market_forward_excess_returns']
print(f"Risk-Free Rate ↔ Excess Returns:      {rf_ex_corr:+.4f}")

print(f"\n📋 CHART DESCRIPTIONS:")
print("=" * 30)
print("Chart 1 (Top-Left):    Scatter plot showing relationship between Forward Returns and Risk-Free Rate")
print("                       Each dot represents one trading day. Red line shows the trend.")
print("")
print("Chart 2 (Top-Center):  Scatter plot of Forward Returns vs Excess Returns")  
print("                       Shows how total stock returns relate to risk-adjusted returns.")
print("")
print("Chart 3 (Top-Right):   Scatter plot of Risk-Free Rate vs Excess Returns")
print("                       Reveals whether risk premiums change with interest rate levels.")
print("")
print("Chart 4 (Bottom-Left): Correlation heatmap with color coding")
print("                       Red = negative correlation, Blue = positive, White = no correlation.")
print("")
print("Chart 5 (Bottom-Center): Statistical summary comparing means and volatilities")
print("                         Blue bars = average returns, Red bars = volatility (risk).")
print("")
print("Chart 6 (Bottom-Right): Rolling 1-year correlations over time")
print("                        Shows how relationships change across different market periods.")

# Create comprehensive visualization with better spacing
fig = make_subplots(
    rows=2, cols=3,
    subplot_titles=[
        '💰 Forward Returns vs Risk-Free Rate',
        '📈 Forward Returns vs Excess Returns', 
        '🏛️ Risk-Free Rate vs Excess Returns',
        '🔥 Correlation Heatmap',
        '📊 Statistical Summary',
        '⏰ Correlation Over Time'
    ],
    specs=[
        [{"type": "scatter"}, {"type": "scatter"}, {"type": "scatter"}],
        [{"type": "heatmap"}, {"type": "bar"}, {"type": "scatter"}]
    ],
    horizontal_spacing=0.12,  # Increased horizontal spacing
    vertical_spacing=0.15     # Increased vertical spacing
)

# 1. Scatter plot: Forward Returns vs Risk-Free Rate
sample_size = min(5000, len(target_data))  # Sample for performance
sample_indices = np.random.choice(len(target_data), sample_size, replace=False)
sample_data = target_data.iloc[sample_indices]

fig.add_trace(
    go.Scatter(
        x=sample_data['risk_free_rate'] * 100,
        y=sample_data['forward_returns'] * 100,
        mode='markers',
        marker=dict(size=3, opacity=0.6, color='blue'),
        name='Data Points',
        hovertemplate='Risk-Free: %{x:.3f}%<br>Forward Ret: %{y:.3f}%<extra></extra>'
    ),
    row=1, col=1
)

# Add trend line
z = np.polyfit(sample_data['risk_free_rate'], sample_data['forward_returns'], 1)
p = np.poly1d(z)
x_trend = np.linspace(sample_data['risk_free_rate'].min(), sample_data['risk_free_rate'].max(), 100)
fig.add_trace(
    go.Scatter(
        x=x_trend * 100,
        y=p(x_trend) * 100,
        mode='lines',
        line=dict(color='red', width=2),
        name=f'Trend (r={fr_rf_corr:.3f})',
        showlegend=False
    ),
    row=1, col=1
)

# 2. Scatter plot: Forward Returns vs Excess Returns
fig.add_trace(
    go.Scatter(
        x=sample_data['market_forward_excess_returns'] * 100,
        y=sample_data['forward_returns'] * 100,
        mode='markers',
        marker=dict(size=3, opacity=0.6, color='green'),
        name='Data Points',
        hovertemplate='Excess Ret: %{x:.3f}%<br>Forward Ret: %{y:.3f}%<extra></extra>'
    ),
    row=1, col=2
)

# Add trend line
z = np.polyfit(sample_data['market_forward_excess_returns'], sample_data['forward_returns'], 1)
p = np.poly1d(z)
x_trend = np.linspace(sample_data['market_forward_excess_returns'].min(), 
                     sample_data['market_forward_excess_returns'].max(), 100)
fig.add_trace(
    go.Scatter(
        x=x_trend * 100,
        y=p(x_trend) * 100,
        mode='lines',
        line=dict(color='red', width=2),
        name=f'Trend (r={fr_ex_corr:.3f})',
        showlegend=False
    ),
    row=1, col=2
)

# 3. Scatter plot: Risk-Free Rate vs Excess Returns
fig.add_trace(
    go.Scatter(
        x=sample_data['risk_free_rate'] * 100,
        y=sample_data['market_forward_excess_returns'] * 100,
        mode='markers',
        marker=dict(size=3, opacity=0.6, color='orange'),
        name='Data Points',
        hovertemplate='Risk-Free: %{x:.3f}%<br>Excess Ret: %{y:.3f}%<extra></extra>'
    ),
    row=1, col=3
)

# Add trend line
z = np.polyfit(sample_data['risk_free_rate'], sample_data['market_forward_excess_returns'], 1)
p = np.poly1d(z)
x_trend = np.linspace(sample_data['risk_free_rate'].min(), sample_data['risk_free_rate'].max(), 100)
fig.add_trace(
    go.Scatter(
        x=x_trend * 100,
        y=p(x_trend) * 100,
        mode='lines',
        line=dict(color='red', width=2),
        name=f'Trend (r={rf_ex_corr:.3f})',
        showlegend=False
    ),
    row=1, col=3
)

# 4. Correlation Heatmap
fig.add_trace(
    go.Heatmap(
        z=target_corr_matrix.values,
        x=target_labels,
        y=target_labels,
        colorscale='RdBu',
        zmid=0,
        text=target_corr_matrix.round(4).values,
        texttemplate='%{text}',
        textfont={"size": 14},
        hovertemplate='%{y} vs %{x}<br>Correlation: %{z:.4f}<extra></extra>',
        showscale=True,
        colorbar=dict(title="Correlation", len=0.4, y=0.2)
    ),
    row=2, col=1
)

# 5. Statistical Summary Bar Chart
stats_data = []
for i, target in enumerate(target_vars):
    target_clean = target_data[target].dropna()
    stats_data.append({
        'Variable': target_labels[i],
        'Mean': target_clean.mean() * 100,
        'Std Dev': target_clean.std() * 100,
        'Skewness': stats.skew(target_clean),
        'Kurtosis': stats.kurtosis(target_clean)
    })

stats_df = pd.DataFrame(stats_data)

# Create side-by-side bars for Mean and Std Dev
x_positions = np.arange(len(stats_df))
width = 0.35

fig.add_trace(
    go.Bar(
        x=x_positions - width/2,
        y=stats_df['Mean'],
        name='Mean (%)',
        marker_color='lightblue',
        width=width,
        hovertemplate='%{x}<br>Mean: %{y:.3f}%<extra></extra>',
        text=[f'{val:.3f}%' for val in stats_df['Mean']],
        textposition='outside'
    ),
    row=2, col=2
)

fig.add_trace(
    go.Bar(
        x=x_positions + width/2,
        y=stats_df['Std Dev'],
        name='Std Dev (%)',
        marker_color='lightcoral',
        width=width,
        hovertemplate='%{x}<br>Std Dev: %{y:.3f}%<extra></extra>',
        text=[f'{val:.3f}%' for val in stats_df['Std Dev']],
        textposition='outside'
    ),
    row=2, col=2
)

# 6. Rolling correlation over time
window_size = 252  # 1 year rolling window
rolling_corr_data = []

# Calculate rolling correlations
for i in range(window_size, len(target_data)):
    window_data = target_data.iloc[i-window_size:i]
    
    rolling_corr_data.append({
        'date_idx': i,
        'fr_rf_corr': window_data['forward_returns'].corr(window_data['risk_free_rate']),
        'fr_ex_corr': window_data['forward_returns'].corr(window_data['market_forward_excess_returns']),
        'rf_ex_corr': window_data['risk_free_rate'].corr(window_data['market_forward_excess_returns'])
    })

rolling_df = pd.DataFrame(rolling_corr_data)

fig.add_trace(
    go.Scatter(
        x=rolling_df['date_idx'],
        y=rolling_df['fr_rf_corr'],
        mode='lines',
        name='Forward vs Risk-Free',
        line=dict(color='blue', width=2)
    ),
    row=2, col=3
)

fig.add_trace(
    go.Scatter(
        x=rolling_df['date_idx'],
        y=rolling_df['fr_ex_corr'],
        mode='lines',
        name='Forward vs Excess',
        line=dict(color='green', width=2)
    ),
    row=2, col=3
)

fig.add_trace(
    go.Scatter(
        x=rolling_df['date_idx'],
        y=rolling_df['rf_ex_corr'],
        mode='lines',
        name='Risk-Free vs Excess',
        line=dict(color='orange', width=2)
    ),
    row=2, col=3
)

# Update layout with better spacing and larger figure
fig.update_layout(
    height=1200,  # Increased height
    width=1400,   # Increased width
    title_text="🔄 Cross-Sectional Analysis: Target Variable Relationships",
    title_x=0.5,
    title_font_size=18,
    showlegend=True,
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="right",
        x=1
    )
)

# Update axis labels with better formatting
fig.update_xaxes(title_text="Risk-Free Rate (%)", row=1, col=1, title_font_size=12)
fig.update_yaxes(title_text="Forward Returns (%)", row=1, col=1, title_font_size=12)

fig.update_xaxes(title_text="Excess Returns (%)", row=1, col=2, title_font_size=12)
fig.update_yaxes(title_text="Forward Returns (%)", row=1, col=2, title_font_size=12)

fig.update_xaxes(title_text="Risk-Free Rate (%)", row=1, col=3, title_font_size=12)
fig.update_yaxes(title_text="Excess Returns (%)", row=1, col=3, title_font_size=12)

# Update bar chart axes
fig.update_xaxes(
    title_text="Target Variables", 
    row=2, col=2, 
    title_font_size=12,
    tickmode='array',
    tickvals=x_positions,
    ticktext=['Forward<br>Returns', 'Risk-Free<br>Rate', 'Market Excess<br>Returns']
)
fig.update_yaxes(title_text="Value (%)", row=2, col=2, title_font_size=12)

fig.update_xaxes(title_text="Time (Days from Start)", row=2, col=3, title_font_size=12)
fig.update_yaxes(title_text="Rolling Correlation (1-Year Window)", row=2, col=3, title_font_size=12)

# Add horizontal line at zero correlation for rolling chart
fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5, row=2, col=3)

fig.show()

# Detailed analysis
print(f"\n📈 DETAILED RELATIONSHIP ANALYSIS:")
print("=" * 45)

# Interpret correlations
def interpret_correlation(corr):
    abs_corr = abs(corr)
    if abs_corr >= 0.7:
        return "Very Strong"
    elif abs_corr >= 0.5:
        return "Strong"
    elif abs_corr >= 0.3:
        return "Moderate"
    elif abs_corr >= 0.1:
        return "Weak"
    else:
        return "Very Weak/None"

relationships = [
    ("Forward Returns", "Risk-Free Rate", fr_rf_corr),
    ("Forward Returns", "Excess Returns", fr_ex_corr), 
    ("Risk-Free Rate", "Excess Returns", rf_ex_corr)
]

for var1, var2, corr in relationships:
    strength = interpret_correlation(corr)
    direction = "positive" if corr > 0 else "negative"
    
    print(f"\n🔗 {var1} ↔ {var2}:")
    print(f"   Correlation: {corr:+.4f} ({strength} {direction})")
    
    if abs(corr) > 0.1:
        r_squared = corr ** 2
        print(f"   Shared variance: {r_squared:.1%}")
        
        if corr > 0:
            print(f"   ↗️ When {var1} increases, {var2} tends to increase")
        else:
            print(f"   ↘️ When {var1} increases, {var2} tends to decrease")

# Key insights
print(f"\n💡 KEY INSIGHTS:")
print("-" * 20)

print(f"1. **Mathematical Relationship**: Forward Returns ≈ Risk-Free Rate + Excess Returns")
print(f"   This explains the high correlation between Forward Returns and Excess Returns")

if abs(fr_rf_corr) < 0.1:
    print(f"2. **Independence**: Forward Returns and Risk-Free Rate move independently")
    print(f"   This suggests different underlying drivers")

if abs(rf_ex_corr) < 0.1:
    print(f"3. **Risk Premium**: Excess Returns are largely independent of Risk-Free Rate")
    print(f"   Market risk premiums don't systematically vary with interest rates")

print(f"\n🎯 MODELING IMPLICATIONS:")
print("-" * 25)
print(f"• High correlation between Forward and Excess Returns is expected (mathematical)")
print(f"• Low correlation with Risk-Free Rate suggests different prediction strategies needed")
print(f"• Consider predicting Risk-Free Rate and Excess Returns separately, then combine")
print(f"• Risk-Free Rate predictions can inform overall market timing strategies")

print(f"\n📊 SUMMARY STATISTICS:")
print("-" * 25)
for stat in stats_data:
    print(f"{stat['Variable']}:")
    print(f"  Mean: {stat['Mean']:+.3f}% | Std: {stat['Std Dev']:.3f}% | Skew: {stat['Skewness']:+.3f}")

# ===== CELL 5 =====
# CELL 1: Interactive Time Series Plots of Target Variables
# =========================================================

import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px

# Load the data
train_df = pd.read_csv(r"C:\Users\Bhekiz\Documents\School\Independent study\hull-tactical-market-prediction\train.csv")

# Create a date column for better visualization (assuming trading days starting from a base date)
# Using date_id as proxy for time - in real data this would be actual dates
train_df['trading_day'] = pd.to_datetime('2000-01-01') + pd.to_timedelta(train_df['date_id'], unit='D')

# Create interactive subplots
fig = make_subplots(
    rows=4, cols=1,
    subplot_titles=[
        '📈 Forward Returns (Next-Day S&P 500 Returns)',
        '🏛️ Risk-Free Rate (Government Securities)',
        '💰 Market Forward Excess Returns (Risk Premium)',
        '📊 Cumulative Performance Comparison'
    ],
    vertical_spacing=0.08,
    row_heights=[0.25, 0.25, 0.25, 0.25]
)

# 1. Forward Returns
fig.add_trace(
    go.Scatter(
        x=train_df['trading_day'],
        y=train_df['forward_returns'] * 100,  # Convert to percentage
        mode='lines',
        name='Forward Returns',
        line=dict(color='#1f77b4', width=1),
        hovertemplate='<b>Date:</b> %{x}<br><b>Return:</b> %{y:.3f}%<extra></extra>'
    ),
    row=1, col=1
)

# Add zero line for forward returns
fig.add_hline(y=0, line_dash="dash", line_color="red", opacity=0.5, row=1, col=1)

# 2. Risk-Free Rate
fig.add_trace(
    go.Scatter(
        x=train_df['trading_day'],
        y=train_df['risk_free_rate'] * 100,  # Convert to percentage
        mode='lines',
        name='Risk-Free Rate',
        line=dict(color='#ff7f0e', width=1),
        hovertemplate='<b>Date:</b> %{x}<br><b>Rate:</b> %{y:.4f}%<extra></extra>'
    ),
    row=2, col=1
)

# 3. Market Forward Excess Returns
fig.add_trace(
    go.Scatter(
        x=train_df['trading_day'],
        y=train_df['market_forward_excess_returns'] * 100,  # Convert to percentage
        mode='lines',
        name='Excess Returns',
        line=dict(color='#2ca02c', width=1),
        hovertemplate='<b>Date:</b> %{x}<br><b>Excess Return:</b> %{y:.3f}%<extra></extra>'
    ),
    row=3, col=1
)

# Add zero line for excess returns
fig.add_hline(y=0, line_dash="dash", line_color="red", opacity=0.5, row=3, col=1)

# 4. Cumulative Performance Comparison
# Calculate cumulative returns for all three series
cum_forward = (1 + train_df['forward_returns'].fillna(0)).cumprod() - 1
cum_risk_free = (1 + train_df['risk_free_rate'].fillna(0)).cumprod() - 1
cum_excess = (1 + train_df['market_forward_excess_returns'].fillna(0)).cumprod() - 1

fig.add_trace(
    go.Scatter(
        x=train_df['trading_day'],
        y=cum_forward * 100,
        mode='lines',
        name='Cumulative Market Returns',
        line=dict(color='#1f77b4', width=2),
        hovertemplate='<b>Date:</b> %{x}<br><b>Cumulative Return:</b> %{y:.1f}%<extra></extra>'
    ),
    row=4, col=1
)

fig.add_trace(
    go.Scatter(
        x=train_df['trading_day'],
        y=cum_risk_free * 100,
        mode='lines',
        name='Cumulative Risk-Free Returns',
        line=dict(color='#ff7f0e', width=2),
        hovertemplate='<b>Date:</b> %{x}<br><b>Cumulative Return:</b> %{y:.1f}%<extra></extra>'
    ),
    row=4, col=1
)

fig.add_trace(
    go.Scatter(
        x=train_df['trading_day'],
        y=cum_excess * 100,
        mode='lines',
        name='Cumulative Excess Returns',
        line=dict(color='#2ca02c', width=2),
        hovertemplate='<b>Date:</b> %{x}<br><b>Cumulative Excess:</b> %{y:.1f}%<extra></extra>'
    ),
    row=4, col=1
)

# Update layout
fig.update_layout(
    height=1200,
    title_text="🎯 Interactive Time Series Analysis of Target Variables",
    title_x=0.5,
    title_font_size=16,
    showlegend=False,
    hovermode='x unified'
)

# Update y-axis labels
fig.update_yaxes(title_text="Daily Return (%)", row=1, col=1)
fig.update_yaxes(title_text="Risk-Free Rate (%)", row=2, col=1)
fig.update_yaxes(title_text="Excess Return (%)", row=3, col=1)
fig.update_yaxes(title_text="Cumulative Return (%)", row=4, col=1)

# Update x-axis labels
fig.update_xaxes(title_text="Trading Day", row=4, col=1)

# Add range selector for better interactivity
fig.update_layout(
    xaxis4=dict(
        rangeselector=dict(
            buttons=list([
                dict(count=252, label="1Y", step="day", stepmode="backward"),
                dict(count=252*3, label="3Y", step="day", stepmode="backward"),
                dict(count=252*5, label="5Y", step="day", stepmode="backward"),
                dict(count=252*10, label="10Y", step="day", stepmode="backward"),
                dict(step="all", label="All")
            ])
        ),
        rangeslider=dict(visible=True),
        type="date"
    )
)

fig.show()

# Display summary statistics
print("\n📊 TARGET VARIABLES SUMMARY STATISTICS")
print("=" * 60)

target_stats = pd.DataFrame({
    'Variable': ['Forward Returns', 'Risk-Free Rate', 'Market Excess Returns'],
    'Mean (%)': [
        train_df['forward_returns'].mean() * 100,
        train_df['risk_free_rate'].mean() * 100,
        train_df['market_forward_excess_returns'].mean() * 100
    ],
    'Std Dev (%)': [
        train_df['forward_returns'].std() * 100,
        train_df['risk_free_rate'].std() * 100,
        train_df['market_forward_excess_returns'].std() * 100
    ],
    'Min (%)': [
        train_df['forward_returns'].min() * 100,
        train_df['risk_free_rate'].min() * 100,
        train_df['market_forward_excess_returns'].min() * 100
    ],
    'Max (%)': [
        train_df['forward_returns'].max() * 100,
        train_df['risk_free_rate'].max() * 100,
        train_df['market_forward_excess_returns'].max() * 100
    ],
    'Total Return (%)': [
        cum_forward.iloc[-1] * 100,
        cum_risk_free.iloc[-1] * 100,
        cum_excess.iloc[-1] * 100
    ],
    'Annualized Return (%)': [
        (cum_forward.iloc[-1] ** (252/len(train_df)) - 1) * 100,
        (cum_risk_free.iloc[-1] ** (252/len(train_df)) - 1) * 100,
        (cum_excess.iloc[-1] ** (252/len(train_df)) - 1) * 100
    ]
})

print(target_stats.round(3))

# ===== CELL 6 =====
pip install scikit-learn

# ===== CELL 7 =====
# CELL 2: Feature-Target Relationship Analysis
# ===========================================

import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
import warnings
warnings.filterwarnings('ignore')

# Check for sklearn availability
try:
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.feature_selection import mutual_info_regression
    SKLEARN_AVAILABLE = True
    print("✅ scikit-learn is available - full analysis enabled")
except ImportError:
    SKLEARN_AVAILABLE = False
    print("⚠️ scikit-learn not found - using correlation analysis only")
    print("   To install: pip install scikit-learn")

# Load data
train_df = pd.read_csv(r"C:\Users\Bhekiz\Documents\School\Independent study\hull-tactical-market-prediction\train.csv")

print("\n🔍 COMPREHENSIVE FEATURE-TARGET RELATIONSHIP ANALYSIS")
print("=" * 65)
print(f"Dataset: {train_df.shape[0]:,} rows × {train_df.shape[1]} columns")

# Define target variables and feature categories
target_vars = ['forward_returns', 'risk_free_rate', 'market_forward_excess_returns']
feature_cols = [col for col in train_df.columns if col not in ['date_id'] + target_vars]

# Categorize features by prefix
feature_categories = {
    'Market Dynamics (M)': [col for col in feature_cols if col.startswith('M')],
    'Macro Economic (E)': [col for col in feature_cols if col.startswith('E')], 
    'Interest Rates (I)': [col for col in feature_cols if col.startswith('I')],
    'Price/Valuation (P)': [col for col in feature_cols if col.startswith('P')],
    'Volatility (V)': [col for col in feature_cols if col.startswith('V')],
    'Sentiment (S)': [col for col in feature_cols if col.startswith('S')],
    'Binary/Dummy (D)': [col for col in feature_cols if col.startswith('D')]
}

print(f"\n📊 FEATURE BREAKDOWN BY CATEGORY:")
for category, features in feature_categories.items():
    print(f"  {category}: {len(features)} features")
print(f"  Total Features: {len(feature_cols)}")

# Calculate correlations for all targets
print(f"\n⚡ Computing correlations for {len(feature_cols)} features...")
correlation_results = {}

for target in target_vars:
    target_data = train_df[target].dropna()
    target_corrs = []
    
    for feature in feature_cols:
        # Get data where both feature and target are available
        valid_mask = train_df[feature].notna() & train_df[target].notna()
        
        if valid_mask.sum() > 100:  # Need at least 100 observations
            corr = train_df.loc[valid_mask, feature].corr(train_df.loc[valid_mask, target])
            if not np.isnan(corr):
                target_corrs.append({
                    'feature': feature,
                    'correlation': corr,
                    'abs_correlation': abs(corr),
                    'data_points': valid_mask.sum(),
                    'category': next((cat for cat, feats in feature_categories.items() 
                                   if feature in feats), 'Unknown')
                })
    
    correlation_results[target] = pd.DataFrame(target_corrs).sort_values(
        'abs_correlation', ascending=False
    )

print("✅ Correlation analysis completed!")

# Define relationship strength categories
def categorize_relationship(abs_corr):
    if abs_corr >= 0.1:
        return "Strong"
    elif abs_corr >= 0.05:
        return "Moderate" 
    elif abs_corr >= 0.02:
        return "Weak"
    else:
        return "Very Weak/None"

# Analyze relationship strengths
relationship_summary = {}
for target, corr_df in correlation_results.items():
    corr_df['relationship_strength'] = corr_df['abs_correlation'].apply(categorize_relationship)
    relationship_summary[target] = corr_df['relationship_strength'].value_counts()

print(f"\n📈 RELATIONSHIP STRENGTH SUMMARY:")
print("=" * 50)
strength_order = ["Strong", "Moderate", "Weak", "Very Weak/None"]

for target in target_vars:
    print(f"\n🎯 {target.replace('_', ' ').title()}:")
    for strength in strength_order:
        count = relationship_summary[target].get(strength, 0)
        pct = (count / len(correlation_results[target])) * 100
        print(f"  {strength:15s}: {count:3d} features ({pct:5.1f}%)")

print(f"\n🔍 CORRELATION THRESHOLDS USED:")
print("  Strong:     |r| ≥ 0.10 (10%+ explained variance)")
print("  Moderate:   |r| ≥ 0.05 (2.5%+ explained variance)")  
print("  Weak:       |r| ≥ 0.02 (0.4%+ explained variance)")
print("  Very Weak:  |r| < 0.02 (minimal predictive value)")

if not SKLEARN_AVAILABLE:
    print(f"\n💡 NOTE: Install scikit-learn to enable Random Forest analysis in CELL 4")
    print("   pip install scikit-learn")

# ===== CELL 8 =====
# CELL 3: Interactive Correlation Heatmaps and Analysis
# =====================================================

# Ensure variables are available (in case cells are run out of order)
if 'target_vars' not in locals():
    target_vars = ['forward_returns', 'risk_free_rate', 'market_forward_excess_returns']
if 'feature_cols' not in locals():
    feature_cols = [col for col in train_df.columns if col not in ['date_id'] + target_vars]
if 'correlation_results' not in locals():
    print("⚠️ Please run CELL 2: 'Feature-Target Relationship Analysis' first!")

# Create interactive correlation heatmaps for each target
fig = make_subplots(
    rows=2, cols=2,
    subplot_titles=[
        '🎯 Forward Returns - Top Correlations by Category',
        '🏛️ Risk-Free Rate - Top Correlations by Category', 
        '💰 Market Excess Returns - Top Correlations by Category',
        '📊 Cross-Target Correlation Comparison'
    ],
    specs=[[{"secondary_y": False}, {"secondary_y": False}],
           [{"secondary_y": False}, {"secondary_y": False}]]
)

# Color scheme for categories
category_colors = {
    'Market Dynamics (M)': '#1f77b4',
    'Macro Economic (E)': '#ff7f0e', 
    'Interest Rates (I)': '#2ca02c',
    'Price/Valuation (P)': '#d62728',
    'Volatility (V)': '#9467bd',
    'Sentiment (S)': '#8c564b',
    'Binary/Dummy (D)': '#e377c2'
}

# Plot top correlations for each target variable
target_positions = [(1,1), (1,2), (2,1)]

for i, target in enumerate(target_vars):
    row, col = target_positions[i]
    
    # Get top 15 correlations for this target
    top_corrs = correlation_results[target].head(15)
    
    # Create bar chart
    colors = [category_colors.get(cat, '#bcbd22') for cat in top_corrs['category']]
    
    fig.add_trace(
        go.Bar(
            x=top_corrs['abs_correlation'],
            y=top_corrs['feature'],
            orientation='h',
            marker_color=colors,
            text=[f"{corr:+.3f}" for corr in top_corrs['correlation']],
            textposition='auto',
            hovertemplate='<b>%{y}</b><br>' +
                         'Correlation: %{text}<br>' +
                         'Category: %{customdata[0]}<br>' +
                         'Data Points: %{customdata[1]:,}<extra></extra>',
            customdata=list(zip(top_corrs['category'], top_corrs['data_points'])),
            showlegend=False
        ),
        row=row, col=col
    )

# Cross-target comparison - show strongest features across all targets
all_features_summary = []
for feature in feature_cols:
    feature_summary = {'feature': feature}
    feature_summary['category'] = next((cat for cat, feats in feature_categories.items() 
                                      if feature in feats), 'Unknown')
    
    for target in target_vars:
        target_corr_df = correlation_results[target]
        feature_row = target_corr_df[target_corr_df['feature'] == feature]
        if len(feature_row) > 0:
            feature_summary[f'{target}_corr'] = feature_row.iloc[0]['correlation']
            feature_summary[f'{target}_abs_corr'] = feature_row.iloc[0]['abs_correlation']
        else:
            feature_summary[f'{target}_corr'] = 0
            feature_summary[f'{target}_abs_corr'] = 0
    
    # Calculate max absolute correlation across all targets
    feature_summary['max_abs_corr'] = max([
        feature_summary[f'{target}_abs_corr'] for target in target_vars
    ])
    
    all_features_summary.append(feature_summary)

all_features_df = pd.DataFrame(all_features_summary).sort_values('max_abs_corr', ascending=False)

# Show top 15 features across all targets
top_cross_features = all_features_df.head(15)

# Create stacked bar for cross-target comparison
fig.add_trace(
    go.Bar(
        x=top_cross_features['forward_returns_abs_corr'],
        y=top_cross_features['feature'],
        orientation='h',
        name='Forward Returns',
        marker_color='#1f77b4',
        opacity=0.8
    ),
    row=2, col=2
)

fig.add_trace(
    go.Bar(
        x=top_cross_features['risk_free_rate_abs_corr'],
        y=top_cross_features['feature'],
        orientation='h', 
        name='Risk-Free Rate',
        marker_color='#ff7f0e',
        opacity=0.8
    ),
    row=2, col=2
)

fig.add_trace(
    go.Bar(
        x=top_cross_features['market_forward_excess_returns_abs_corr'],
        y=top_cross_features['feature'],
        orientation='h',
        name='Excess Returns',
        marker_color='#2ca02c',
        opacity=0.8
    ),
    row=2, col=2
)

# Update layout
fig.update_layout(
    height=1000,
    title_text="🔗 Feature-Target Correlation Analysis Dashboard",
    title_x=0.5,
    showlegend=True,
    barmode='group'
)

# Update axes
for i in range(1, 4):
    row, col = target_positions[i-1] if i <= 3 else (2,2)
    fig.update_xaxes(title_text="Absolute Correlation", row=row, col=col)
    fig.update_yaxes(title_text="Features", row=row, col=col)

fig.update_xaxes(title_text="Absolute Correlation", row=2, col=2)
fig.update_yaxes(title_text="Top Features", row=2, col=2)

fig.show()

# Print detailed analysis
print("\n" + "="*70)
print("🎯 TOP FEATURES BY TARGET VARIABLE")
print("="*70)

for target in target_vars:
    print(f"\n📈 {target.replace('_', ' ').title().upper()}:")
    print("-" * 50)
    
    top_10 = correlation_results[target].head(10)
    print(f"{'Rank':<4} {'Feature':<12} {'Correlation':<12} {'Category':<20} {'Strength':<12}")
    print("-" * 70)
    
    for idx, (_, row) in enumerate(top_10.iterrows(), 1):
        strength = categorize_relationship(row['abs_correlation'])
        print(f"{idx:<4} {row['feature']:<12} {row['correlation']:+8.4f}    {row['category']:<20} {strength:<12}")

print(f"\n" + "="*70)
print("🏆 OVERALL STRONGEST FEATURES (ACROSS ALL TARGETS)")
print("="*70)

print(f"{'Rank':<4} {'Feature':<12} {'Max |r|':<10} {'Category':<20} {'Best Target':<15}")
print("-" * 75)

for idx, (_, row) in enumerate(all_features_df.head(15).iterrows(), 1):
    # Find which target has the highest correlation
    target_corrs = {target: abs(row[f'{target}_corr']) for target in target_vars}
    best_target = max(target_corrs, key=target_corrs.get)
    best_corr = row[f'{best_target}_corr']
    
    print(f"{idx:<4} {row['feature']:<12} {row['max_abs_corr']:.4f}     {row['category']:<20} {best_target:<15}")

print(f"\n🔍 KEY INSIGHTS:")
print(f"• Most predictive features have correlations < 0.1 (typical for financial markets)")
print(f"• {target_vars[0]} shows strongest relationships with {correlation_results[target_vars[0]].iloc[0]['category']}")
print(f"• Feature availability varies significantly across time periods")
print(f"• Binary/dummy variables show the most consistent data availability")

# ===== CELL 9 =====
# CELL 4: Feature Importance Analysis (Alternative Methods)
# ========================================================

# Ensure variables are available (in case cells are run out of order)
if 'target_vars' not in locals():
    target_vars = ['forward_returns', 'risk_free_rate', 'market_forward_excess_returns']
if 'feature_cols' not in locals():
    feature_cols = [col for col in train_df.columns if col not in ['date_id'] + target_vars]
if 'feature_categories' not in locals():
    feature_categories = {
        'Market Dynamics (M)': [col for col in feature_cols if col.startswith('M')],
        'Macro Economic (E)': [col for col in feature_cols if col.startswith('E')], 
        'Interest Rates (I)': [col for col in feature_cols if col.startswith('I')],
        'Price/Valuation (P)': [col for col in feature_cols if col.startswith('P')],
        'Volatility (V)': [col for col in feature_cols if col.startswith('V')],
        'Sentiment (S)': [col for col in feature_cols if col.startswith('S')],
        'Binary/Dummy (D)': [col for col in feature_cols if col.startswith('D')]
    }
if 'category_colors' not in locals():
    category_colors = {
        'Market Dynamics (M)': '#1f77b4',
        'Macro Economic (E)': '#ff7f0e', 
        'Interest Rates (I)': '#2ca02c',
        'Price/Valuation (P)': '#d62728',
        'Volatility (V)': '#9467bd',
        'Sentiment (S)': '#8c564b',
        'Binary/Dummy (D)': '#e377c2'
    }

# Check for sklearn availability
try:
    from sklearn.ensemble import RandomForestRegressor
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

if SKLEARN_AVAILABLE:
    print("\n🌲 RANDOM FOREST FEATURE IMPORTANCE ANALYSIS")
    print("=" * 55)

    # Use only recent data with minimal missing values (last 25% of dataset)
    recent_data_start = int(len(train_df) * 0.75)
    recent_df = train_df.iloc[recent_data_start:].copy()

    print(f"Using recent data: {len(recent_df):,} observations (last 25% of dataset)")
    print(f"Date range: {recent_df['date_id'].min()} to {recent_df['date_id'].max()}")

    # Prepare features with minimal missing data
    feature_completeness = recent_df[feature_cols].notna().mean()
    good_features = feature_completeness[feature_completeness >= 0.8].index.tolist()  # ≥80% complete

    print(f"Features with ≥80% completeness: {len(good_features)}")

    # Create Random Forest analysis for each target
    rf_results = {}
    feature_importance_summary = []

    for target in target_vars:
        print(f"\n🎯 Analyzing {target}...")
        
        # Prepare data
        target_data = recent_df[target].dropna()
        feature_data = recent_df.loc[target_data.index, good_features].fillna(method='ffill').fillna(0)
        
        if len(target_data) > 500:  # Minimum observations needed
            # Train Random Forest
            rf = RandomForestRegressor(
                n_estimators=100,
                max_depth=10, 
                random_state=42,
                n_jobs=-1
            )
            
            rf.fit(feature_data, target_data)
            
            # Get feature importance
            importance_df = pd.DataFrame({
                'feature': good_features,
                'importance': rf.feature_importances_,
                'category': [next((cat for cat, feats in feature_categories.items() 
                                 if feat in feats), 'Unknown') for feat in good_features]
            }).sort_values('importance', ascending=False)
            
            rf_results[target] = importance_df
            
            print(f"  ✅ Completed - R² score: {rf.score(feature_data, target_data):.4f}")
        else:
            print(f"  ❌ Insufficient data: {len(target_data)} observations")

    # Create visualizations if we have results
    if rf_results:
        # Create interactive feature importance visualization
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=[
                '🌲 Forward Returns - Random Forest Importance',
                '🌲 Risk-Free Rate - Random Forest Importance',
                '🌲 Market Excess Returns - Random Forest Importance', 
                '📊 Category-wise Importance Distribution'
            ]
        )

        # Plot Random Forest importance for each target
        target_positions = [(1,1), (1,2), (2,1)]

        for i, target in enumerate(target_vars):
            if target in rf_results:
                row, col = target_positions[i]
                top_features = rf_results[target].head(15)
                
                colors = [category_colors.get(cat, '#bcbd22') for cat in top_features['category']]
                
                fig.add_trace(
                    go.Bar(
                        x=top_features['importance'],
                        y=top_features['feature'],
                        orientation='h',
                        marker_color=colors,
                        text=[f"{imp:.4f}" for imp in top_features['importance']],
                        textposition='auto',
                        hovertemplate='<b>%{y}</b><br>' +
                                     'Importance: %{x:.4f}<br>' +
                                     'Category: %{customdata}<extra></extra>',
                        customdata=top_features['category'],
                        showlegend=False
                    ),
                    row=row, col=col
                )

        # Category-wise importance summary
        category_importance = {}
        for target in target_vars:
            if target in rf_results:
                cat_importance = rf_results[target].groupby('category')['importance'].sum().sort_values(ascending=False)
                category_importance[target] = cat_importance

        if category_importance:
            # Get all categories
            all_categories = set()
            for target_cats in category_importance.values():
                all_categories.update(target_cats.index)
            
            all_categories = sorted(list(all_categories))
            
            # Create grouped bar chart for categories
            for i, target in enumerate(target_vars):
                if target in category_importance:
                    values = [category_importance[target].get(cat, 0) for cat in all_categories]
                    
                    fig.add_trace(
                        go.Bar(
                            x=all_categories,
                            y=values,
                            name=target.replace('_', ' ').title(),
                            marker_color=['#1f77b4', '#ff7f0e', '#2ca02c'][i]
                        ),
                        row=2, col=2
                    )

        # Update layout
        fig.update_layout(
            height=1000,
            title_text="🌲 Random Forest Feature Importance Analysis",
            title_x=0.5,
            showlegend=True
        )

        # Update axes
        for i in range(1, 4):
            row, col = target_positions[i-1]
            fig.update_xaxes(title_text="Feature Importance", row=row, col=col)
            fig.update_yaxes(title_text="Features", row=row, col=col)

        fig.update_xaxes(title_text="Feature Categories", row=2, col=2)
        fig.update_yaxes(title_text="Total Importance", row=2, col=2)

        fig.show()

        # Print Random Forest results
        print(f"\n🏆 TOP RANDOM FOREST FEATURES BY TARGET:")
        print("=" * 60)

        for target in target_vars:
            if target in rf_results:
                print(f"\n📈 {target.replace('_', ' ').title().upper()}:")
                print("-" * 50)
                
                top_10 = rf_results[target].head(10)
                print(f"{'Rank':<4} {'Feature':<12} {'Importance':<12} {'Category':<20}")
                print("-" * 60)
                
                for idx, (_, row) in enumerate(top_10.iterrows(), 1):
                    print(f"{idx:<4} {row['feature']:<12} {row['importance']:8.4f}    {row['category']:<20}")
    else:
        print("❌ No Random Forest results - insufficient data in recent period")
        
else:
    print("\n📊 ALTERNATIVE FEATURE IMPORTANCE ANALYSIS")
    print("=" * 50)
    print("⚠️ scikit-learn not available - using correlation-based importance")
    
    # Alternative: Use correlation strength as importance measure
    rf_results = {}
    
    for target in target_vars:
        if target in correlation_results:
            # Convert correlation to "importance" (normalized absolute correlation)
            corr_data = correlation_results[target].copy()
            max_corr = corr_data['abs_correlation'].max()
            
            # Normalize to 0-1 scale like Random Forest importance
            corr_data['importance'] = corr_data['abs_correlation'] / max_corr if max_corr > 0 else 0
            
            rf_results[target] = corr_data.sort_values('importance', ascending=False)
    
    # Create visualization using correlation-based importance
    if rf_results:
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=[
                '📈 Forward Returns - Correlation-Based Importance',
                '📈 Risk-Free Rate - Correlation-Based Importance',
                '📈 Market Excess Returns - Correlation-Based Importance',
                '📊 Category-wise Importance Distribution'
            ]
        )

        target_positions = [(1,1), (1,2), (2,1)]

        for i, target in enumerate(target_vars):
            if target in rf_results:
                row, col = target_positions[i]
                top_features = rf_results[target].head(15)
                
                colors = [category_colors.get(cat, '#bcbd22') for cat in top_features['category']]
                
                fig.add_trace(
                    go.Bar(
                        x=top_features['importance'],
                        y=top_features['feature'],
                        orientation='h',
                        marker_color=colors,
                        text=[f"{imp:.3f}" for imp in top_features['importance']],
                        textposition='auto',
                        hovertemplate='<b>%{y}</b><br>' +
                                     'Norm. |Correlation|: %{x:.3f}<br>' +
                                     'Category: %{customdata}<extra></extra>',
                        customdata=top_features['category'],
                        showlegend=False
                    ),
                    row=row, col=col
                )

        # Category-wise importance
        category_importance = {}
        for target in target_vars:
            if target in rf_results:
                cat_importance = rf_results[target].groupby('category')['importance'].sum().sort_values(ascending=False)
                category_importance[target] = cat_importance

        if category_importance:
            all_categories = set()
            for target_cats in category_importance.values():
                all_categories.update(target_cats.index)
            
            all_categories = sorted(list(all_categories))
            
            for i, target in enumerate(target_vars):
                if target in category_importance:
                    values = [category_importance[target].get(cat, 0) for cat in all_categories]
                    
                    fig.add_trace(
                        go.Bar(
                            x=all_categories,
                            y=values,
                            name=target.replace('_', ' ').title(),
                            marker_color=['#1f77b4', '#ff7f0e', '#2ca02c'][i]
                        ),
                        row=2, col=2
                    )

        fig.update_layout(
            height=1000,
            title_text="📈 Correlation-Based Feature Importance Analysis",
            title_x=0.5,
            showlegend=True
        )

        for i in range(1, 4):
            row, col = target_positions[i-1]
            fig.update_xaxes(title_text="Normalized |Correlation|", row=row, col=col)
            fig.update_yaxes(title_text="Features", row=row, col=col)

        fig.update_xaxes(title_text="Feature Categories", row=2, col=2)
        fig.update_yaxes(title_text="Total Importance", row=2, col=2)

        fig.show()

        print(f"\n📈 TOP FEATURES BY CORRELATION-BASED IMPORTANCE:")
        print("=" * 60)

        for target in target_vars:
            if target in rf_results:
                print(f"\n📊 {target.replace('_', ' ').title().upper()}:")
                print("-" * 50)
                
                top_10 = rf_results[target].head(10)
                print(f"{'Rank':<4} {'Feature':<12} {'Norm.|r|':<12} {'Category':<20}")
                print("-" * 60)
                
                for idx, (_, row) in enumerate(top_10.iterrows(), 1):
                    print(f"{idx:<4} {row['feature']:<12} {row['importance']:8.3f}    {row['category']:<20}")

    print(f"\n💡 TO ENABLE RANDOM FOREST ANALYSIS:")
    print("   Run: pip install scikit-learn")
    print("   Then re-run this cell for advanced feature importance analysis")

print(f"\n🔍 ANALYSIS INSIGHTS:")
method = "Random Forest" if SKLEARN_AVAILABLE else "Correlation-based"
print(f"• {method} importance analysis completed")
print(f"• Higher importance = more useful for prediction")
if SKLEARN_AVAILABLE:
    print(f"• Random Forest captures non-linear relationships")
    print(f"• Results based on recent data with minimal missing values")
else:
    print(f"• Correlation captures linear relationships only")
    print(f"• Install scikit-learn for non-linear analysis")

# ===== CELL 10 =====
# CELL 5: Final Analysis Summary and Feature Selection Recommendations
# ====================================================================

# Ensure variables are available (in case cells are run out of order)
if 'target_vars' not in locals():
    target_vars = ['forward_returns', 'risk_free_rate', 'market_forward_excess_returns']
if 'feature_categories' not in locals():
    feature_categories = {
        'Market Dynamics (M)': [col for col in train_df.columns if col.startswith('M') and col not in ['date_id'] + target_vars],
        'Macro Economic (E)': [col for col in train_df.columns if col.startswith('E') and col not in ['date_id'] + target_vars], 
        'Interest Rates (I)': [col for col in train_df.columns if col.startswith('I') and col not in ['date_id'] + target_vars],
        'Price/Valuation (P)': [col for col in train_df.columns if col.startswith('P') and col not in ['date_id'] + target_vars],
        'Volatility (V)': [col for col in train_df.columns if col.startswith('V') and col not in ['date_id'] + target_vars],
        'Sentiment (S)': [col for col in train_df.columns if col.startswith('S') and col not in ['date_id'] + target_vars],
        'Binary/Dummy (D)': [col for col in train_df.columns if col.startswith('D') and col not in ['date_id'] + target_vars]
    }

if 'correlation_results' not in locals() or 'rf_results' not in locals():
    print("⚠️ Please run the previous analysis cells first!")
    print("Required: CELL 2 (Feature-Target Relationship Analysis) & CELL 4 (Random Forest Analysis)")
else:
    print("\n" + "="*80)
    print("🎯 COMPREHENSIVE FEATURE-TARGET RELATIONSHIP SUMMARY")
    print("="*80)

    # Combine correlation and Random Forest insights
    print(f"\n📊 METHODOLOGY COMPARISON:")
    print("-" * 40)
    print(f"• Correlation Analysis: Linear relationships, all available data")
    print(f"• Random Forest: Non-linear relationships, recent complete data only")
    print(f"• Both methods identify different aspects of predictive value")

    # Create master feature ranking
    master_features = {}

    # Add correlation rankings (weighted by strength)
    for target in target_vars:
        if target in correlation_results:
            for idx, (_, row) in enumerate(correlation_results[target].head(20).iterrows()):
                feature = row['feature']
                if feature not in master_features:
                    master_features[feature] = {
                        'feature': feature,
                        'category': row['category'],
                        'correlation_rank': {},
                        'rf_rank': {},
                        'correlation_strength': {},
                        'rf_importance': {}
                    }
                
                master_features[feature]['correlation_rank'][target] = idx + 1
                master_features[feature]['correlation_strength'][target] = abs(row['correlation'])

    # Add Random Forest rankings
    for target in target_vars:
        if target in rf_results:
            for idx, (_, row) in enumerate(rf_results[target].head(20).iterrows()):
                feature = row['feature']
                if feature not in master_features:
                    master_features[feature] = {
                        'feature': feature,
                        'category': row['category'],
                        'correlation_rank': {},
                        'rf_rank': {},
                        'correlation_strength': {},
                        'rf_importance': {}
                    }
                
                master_features[feature]['rf_rank'][target] = idx + 1
                master_features[feature]['rf_importance'][target] = row['importance']

    # Calculate composite scores
    for feature_name, feature_data in master_features.items():
        # Correlation score (average across targets, higher = better)
        corr_scores = list(feature_data['correlation_strength'].values())
        feature_data['avg_correlation'] = np.mean(corr_scores) if corr_scores else 0
        
        # Random Forest score (average across targets, higher = better)
        rf_scores = list(feature_data['rf_importance'].values())
        feature_data['avg_rf_importance'] = np.mean(rf_scores) if rf_scores else 0
        
        # Combined score (normalized and weighted)
        norm_corr = feature_data['avg_correlation'] / 0.1  # Normalize by strong threshold
        norm_rf = feature_data['avg_rf_importance'] / 0.05  # Normalize by typical importance
        
        feature_data['composite_score'] = (norm_corr + norm_rf) / 2
        
        # Count targets where feature appears in top rankings
        feature_data['target_coverage'] = len(set(
            list(feature_data['correlation_rank'].keys()) + 
            list(feature_data['rf_rank'].keys())
        ))

    # Sort by composite score
    master_ranking = sorted(master_features.values(), 
                           key=lambda x: x['composite_score'], reverse=True)

    # Categorize features by strength
    strong_features = []
    moderate_features = []
    weak_features = []
    no_relationship = []

    for feature_data in master_ranking:
        max_corr = feature_data['avg_correlation']
        max_rf = feature_data['avg_rf_importance']
        
        if max_corr >= 0.05 or max_rf >= 0.02:
            strong_features.append(feature_data)
        elif max_corr >= 0.03 or max_rf >= 0.01:
            moderate_features.append(feature_data)
        elif max_corr >= 0.01 or max_rf >= 0.005:
            weak_features.append(feature_data)
        else:
            no_relationship.append(feature_data)

    print(f"\n🏆 FINAL FEATURE CATEGORIZATION:")
    print("=" * 50)
    print(f"Strong Relationship:    {len(strong_features):3d} features (recommended for modeling)")
    print(f"Moderate Relationship:  {len(moderate_features):3d} features (consider for ensemble)")
    print(f"Weak Relationship:      {len(weak_features):3d} features (use cautiously)")
    print(f"No/Minimal Relationship: {len(no_relationship):3d} features (likely not useful)")

    print(f"\n🥇 TOP 20 FEATURES (COMBINED RANKING):")
    print("=" * 80)
    print(f"{'Rank':<4} {'Feature':<12} {'Category':<20} {'Avg |r|':<10} {'Avg RF':<10} {'Targets':<8} {'Score':<8}")
    print("-" * 80)

    for rank, feature_data in enumerate(master_ranking[:20], 1):
        print(f"{rank:<4} {feature_data['feature']:<12} {feature_data['category']:<20} "
              f"{feature_data['avg_correlation']:.4f}     {feature_data['avg_rf_importance']:.4f}     "
              f"{feature_data['target_coverage']:<8} {feature_data['composite_score']:.3f}")

    # Category-wise recommendations
    print(f"\n📋 FEATURE SELECTION RECOMMENDATIONS BY CATEGORY:")
    print("=" * 60)

    category_recommendations = {}
    for feature_data in strong_features + moderate_features:
        category = feature_data['category']
        if category not in category_recommendations:
            category_recommendations[category] = []
        category_recommendations[category].append(feature_data)

    for category, features in category_recommendations.items():
        features_sorted = sorted(features, key=lambda x: x['composite_score'], reverse=True)
        print(f"\n📊 {category}:")
        print(f"  Recommended features: {len(features_sorted)}")
        print(f"  Top 3: {', '.join([f['feature'] for f in features_sorted[:3]])}")

    # Data availability analysis
    print(f"\n📅 DATA AVAILABILITY ANALYSIS:")
    print("=" * 40)

    # Calculate data availability for top features
    for period_name, period_slice in [
        ("Early Period (0-25%)", slice(0, len(train_df)//4)),
        ("Recent Period (75-100%)", slice(3*len(train_df)//4, len(train_df)))
    ]:
        print(f"\n{period_name}:")
        
        period_data = train_df.iloc[period_slice]
        top_features_list = [f['feature'] for f in master_ranking[:20]]
        availability = period_data[top_features_list].notna().mean().sort_values(ascending=False)
        
        print(f"  Best availability: {availability.iloc[0]:.1%} ({availability.index[0]})")
        print(f"  Worst availability: {availability.iloc[-1]:.1%} ({availability.index[-1]})")
        print(f"  Average availability: {availability.mean():.1%}")

    print(f"\n🎯 MODELING RECOMMENDATIONS:")
    print("=" * 40)
    print(f"1. START WITH: Top {len(strong_features)} strong features for baseline models")
    print(f"2. EXPAND TO: Include {len(moderate_features)} moderate features for ensemble methods")  
    print(f"3. FEATURE ENGINEERING: Create combinations of top features within categories")
    print(f"4. TIME PERIODS: Consider separate models for early vs recent periods")
    print(f"5. MISSING DATA: Use forward-fill or interpolation for top features")

    print(f"\n💡 KEY INSIGHTS:")
    print("=" * 25)
    print(f"• Financial relationships are generally weak (correlations < 0.1)")
    if feature_categories['Binary/Dummy (D)']:
        print(f"• {feature_categories['Binary/Dummy (D)'][0][:2]}-category features have best data availability")
    print(f"• Random Forest identifies different features than correlation analysis")
    print(f"• Feature importance varies significantly across target variables")
    print(f"• Recent data has much better feature completeness than early periods")

    print(f"\n📈 NEXT STEPS:")
    print("=" * 20)
    print(f"1. Build baseline models using top 10-15 strong features")
    print(f"2. Test feature combinations and interactions")
    print(f"3. Implement time-aware validation (no data leakage)")
    print(f"4. Consider regime-specific models")
    print(f"5. Focus on directional accuracy over magnitude prediction")

    print(f"\n" + "="*80)
    print("🚀 READY FOR MODEL DEVELOPMENT!")
    print("="*80)

# ===== CELL 11 =====


