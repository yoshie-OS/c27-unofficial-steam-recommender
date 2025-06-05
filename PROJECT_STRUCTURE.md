# Game Recommendation Webapp

## Project Structure

```
webapp/
├── api/
│   ├── connector.py          # Database/API connection handlers
│   └── utils.py              # Utility functions for API operations
├── data/
│   ├── filtered_tags.csv     # Preprocessed game tags dataset
│   └── games_filtered.csv    # Filtered games dataset
├── evaluation/
│   └── metrics.py            # Model evaluation and performance metrics
├── recommender/
│   ├── model.py              # Core recommendation algorithm/model
│   └── preprocessing.py      # Data preprocessing and feature engineering
├── requirements.txt          # Python dependencies
├── readme.md                 # Project documentation
└── streamlit_app.py          # Main Streamlit application entry point
```

## Directory Descriptions

### `/api/`
Contains modules for external connections and API integrations:
- **connector.py**: Handles database connections, API endpoints, and data retrieval
- **utils.py**: Helper functions for API operations, data formatting, and response handling

### `/data/`
Stores preprocessed datasets used by the recommendation system:
- **filtered_tags.csv**: Clean game tags data with relevant categories
- **games_filtered.csv**: Curated games dataset with features for recommendations

### `/evaluation/`
Houses model evaluation and testing components:
- **metrics.py**: Performance metrics, validation functions, and recommendation quality assessment

### `/recommender/`
Core recommendation engine components:
- **model.py**: Main recommendation algorithms and model implementations
- **preprocessing.py**: Data cleaning, feature extraction, and transformation pipeline

### Root Files
- **streamlit_app.py**: Main application interface and user interaction logic
- **requirements.txt**: Project dependencies and package versions
- **readme.md**: Project documentation, setup instructions, and usage guide
