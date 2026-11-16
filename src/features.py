# The exact columns your model uses to make predictions
FEATURES = (
    'hotel',
    'distribution_channel',
    'is_repeated_guest',
    'previous_cancellations',
    'previous_bookings_not_canceled'
)

# The list of hotels in the dataset
HOTELS = ['Resort Hotel', 'City Hotel']

def predictors(rows):
    """
    The backtesting script calls this to let you engineer new features before scoring.
    Since your XGBoost model uses standard columns, we simply pass.
    """
    pass
