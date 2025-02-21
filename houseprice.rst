.. code:: ipython3

    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import mean_absolute_error
    
    # Ignore warnings
    import warnings
    warnings.filterwarnings("ignore")
    
    print("Libraries loaded successfully!")
    


.. parsed-literal::

    Libraries loaded successfully!
    

.. code:: ipython3

    # Creating a sample dataset
    data = {
        'Area (sq ft)': [1000, 1500, 2000, 2500, 3000, 3500, 4000, 5000],
        'Bedrooms': [2, 3, 3, 4, 4, 5, 5,5],
        'Age (years)': [5, 10, 15, 20, 25, 30, 35,75],
        'Price (in lakhs)': [50, 70, 90, 110, 130, 150, 170,34]
    }
    
    df = pd.DataFrame(data)
    print(df)
    


.. parsed-literal::

       Area (sq ft)  Bedrooms  Age (years)  Price (in lakhs)
    0          1000         2            5                50
    1          1500         3           10                70
    2          2000         3           15                90
    3          2500         4           20               110
    4          3000         4           25               130
    5          3500         5           30               150
    6          4000         5           35               170
    7          5000         5           75                34
    

.. code:: ipython3

    # Splitting data into input (X) and output (Y)
    X = df[['Area (sq ft)', 'Bedrooms', 'Age (years)']]
    y = df['Price (in lakhs)']
    
    # Split into 80% training and 20% testing
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print("Training and testing data prepared.")
    


.. parsed-literal::

    Training and testing data prepared.
    

.. code:: ipython3

    # Train the model
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    print("Model trained successfully!")
    


.. parsed-literal::

    Model trained successfully!
    

.. code:: ipython3

    # Predicting values
    y_pred = model.predict(X_test)
    
    
    # Display predictions vs actual prices
    df_results = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
    print(df_results)
    
    # Check error
    error = mean_absolute_error(y_test, y_pred)
    print(f"Mean Absolute Error: {error:.2f} lakhs")
    


.. parsed-literal::

       Actual  Predicted
    1      70       70.0
    5     150      150.0
    Mean Absolute Error: 0.00 lakhs
    

