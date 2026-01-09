# Alt-Data Pulse Dashboard - Plotly Dash Version

Production-ready Bloomberg Terminal-style dashboard for quantitative signal analysis.

## Features

### ✅ Phase 2A-2H Complete

- **Bloomberg Dark Theme**: Professional dark interface matching Bloomberg Terminal aesthetics
- **Interactive Charts**: Plotly time series charts with hover, zoom, and pan capabilities
- **Real-time Data**: Live data from RevenuePredictor with configurable auto-refresh
- **KPI Cards**: Revenue Proxy, Wall Street Consensus, Delta, Signal Strength
- **Multi-Brand Support**: STARBUCKS, MCDONALD'S, CHIPOTLE
- **Date Filtering**: Last 90 Days, 6 Months, Year, or All Available
- **Auto-Refresh**: Configurable (1, 5, 15 minutes, or off)
- **CSV Export**: Download trend data with date filtering
- **Responsive Layout**: Works on desktop and tablet devices

## Quick Start

### Run the Dashboard

```bash
cd /Users/aaron/Desktop/VScode/Alt-Data_Pulse-Dashboard
python dash_app/app.py
```

Then open **http://localhost:8050** in your browser.

### Dependencies

All required packages are already installed:
- `dash >= 2.14.0`
- `dash-bootstrap-components >= 1.5.0`
- `plotly >= 5.18.0`
- `pandas`

## Architecture

```
dash_app/
├── app.py              # Main application entry point
├── utils.py            # Data loading utilities (wraps RevenuePredictor)
├── charts.py           # Plotly chart builders with Bloomberg theme
├── assets/
│   └── styles.css      # Bloomberg Terminal CSS theme
└── README.md           # This file
```

## Key Components

### Data Flow

1. **RevenuePredictor** (`ingest/predictor.py`) - Core quant engine
2. **utils.py** - Data loading layer (singleton predictor instance)
3. **charts.py** - Bloomberg-style Plotly chart generation
4. **app.py** - Dash layout and callbacks

### Main Features

#### KPI Cards
- **Revenue Proxy**: Predicted quarterly revenue from alt-data signals
- **Wall St. Consensus**: LSEG consensus estimates (or historical fallback)
- **Delta vs Consensus**: Beat/Miss percentage
- **Signal Strength**: R² correlation (Very High, High, Medium, Low)

#### Charts
- **Foot Traffic Index**: 7-day rolling average, normalized to mean=100
- **Average Ticket Size**: Consumer Edge spend data, 7-day smoothed
- **Combined View**: Dual y-axis chart showing both signals

#### Controls
- **Brand Selector**: Switch between STARBUCKS, MCDONALD'S, CHIPOTLE
- **Date Range**: Filter charts (Last 90 Days, 6 Months, Year, All Available)
- **Auto-Refresh**: Configurable interval (1, 5, 15 min or off)
- **Manual Refresh**: Immediate data reload button
- **CSV Export**: Download filtered trend data

## Usage Examples

### Basic Usage

1. **Select Brand**: Choose from dropdown (default: STARBUCKS)
2. **View Charts**: Interactive Plotly charts with hover tooltips
3. **Filter Date Range**: Adjust time window for analysis
4. **Export Data**: Click "Export CSV" to download trend data

### Auto-Refresh Configuration

1. Use dropdown to select refresh interval
2. Default: 5 minutes
3. Set to "Off" to disable auto-refresh
4. Use "Refresh" button for manual updates

### Debug Mode

1. Click "▼ Show Debug Data" to expand
2. View raw signal JSON
3. See trend data preview (first 10 rows)
4. Click "▲ Hide Debug Data" to collapse

## Comparison: Streamlit vs Dash

| Feature | Streamlit | Dash |
|---------|-----------|------|
| Chart Library | Altair | Plotly |
| Interactivity | Limited | Full callbacks |
| Auto-refresh | Basic | Configurable intervals |
| Dark Theme | Custom CSS | Built-in + Custom |
| Export | Manual | dcc.Download component |
| Performance | Slower reloads | Fast updates |
| Production Ready | Good | Excellent |

## Customization

### Changing Colors

Edit `dash_app/assets/styles.css`:

```css
:root {
    --accent-green: #3fb950;  /* Positive deltas */
    --accent-red: #f85149;    /* Negative deltas */
    --accent-blue: #58a6ff;   /* Primary accent */
}
```

### Adding New Brands

1. Update `ingest/predictor.py` - Add to `HISTORICAL_REVENUE`
2. Update `dash_app/utils.py` - Add to `get_available_brands()`
3. Refresh dashboard - New brand appears in selector

### Adjusting Refresh Intervals

Edit `dash_app/app.py`:

```python
dcc.Dropdown(
    id="refresh-interval-selector",
    options=[
        {"label": "Auto-refresh: 30 sec", "value": 0.5},  # Add new option
        {"label": "Auto-refresh: Off", "value": 0},
        # ... existing options
    ],
)
```

## Deployment

### Local Development

```bash
python dash_app/app.py
```

### Production with Gunicorn

```bash
pip install gunicorn
cd dash_app
gunicorn app:server -b 0.0.0.0:8050 --workers 4
```

### Docker (Optional)

```dockerfile
FROM python:3.10
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
CMD ["gunicorn", "dash_app.app:server", "-b", "0.0.0.0:8050", "--workers", "4"]
```

## Troubleshooting

### Import Errors

If you see `ModuleNotFoundError: No module named 'ingest'`:

```bash
# Run from project root, not dash_app/
cd /Users/aaron/Desktop/VScode/Alt-Data_Pulse-Dashboard
python dash_app/app.py
```

### Data Not Loading

Check that data files exist:
- `data/clean_spend_daily.csv`
- `data/clean_traffic_daily.csv`
- `data/lseg_raw_estimates.csv` (optional)

Run data pipeline if needed:
```bash
python ingest/pipeline.py --full --start-date 2023-01-01
```

### Charts Not Displaying

1. Check browser console for JavaScript errors
2. Clear browser cache
3. Verify Plotly is installed: `pip show plotly`

## Performance

- **Initial Load**: ~2-3 seconds
- **Brand Switch**: ~500ms
- **Date Range Filter**: ~100ms
- **Auto-Refresh**: Async, non-blocking

## Future Enhancements

Potential additions (not implemented):

- [ ] Multi-monitor layout support
- [ ] Keyboard shortcuts for navigation
- [ ] News feed integration (LSEG news API)
- [ ] PDF report generation
- [ ] Email/Slack alerts for signal changes
- [ ] User authentication (if multi-user)
- [ ] Saved views/preferences
- [ ] Mobile-responsive improvements

## License

Part of the Alt-Data Pulse Dashboard project.

## Support

For issues or questions:
1. Check this README
2. Review `/Users/aaron/Desktop/VScode/Alt-Data_Pulse-Dashboard/.claude/CLAUDE.md`
3. Examine source code comments in `app.py`, `utils.py`, `charts.py`
