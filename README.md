# fos
Forecast the Global Horizontal Irradiance with a stacked regressor using only weather forecast data.


## How to use it

1. Install the needed libraries from requirements.txt with:
```pip install -r requirements.txt```
2. Import the Stacked model from stacked_model file.
3. Either load a pre-trained model using `load_model` or create a new object.
4. Fit the model to your data using the fit method to achieve an higher accuracy (this step can be skipped in case of unavailability of training data).
5. Predict on your dataset calling the predict method.


In the example file is possible to see one implementation of this approach, specifically the one used to forecast the Global Horizontal Irradiance in the area of The Bronx, NY.
