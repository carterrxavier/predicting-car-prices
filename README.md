# See model in application
## [Predicting car prices](https://share.streamlit.io/carterrxavier/predicting-car-prices/main/app.py) 


# Car Price Prediction
## This project is an attempt to gain insight on the price prediction for used cars. This model will help shape a price prediction using a regression machine learning model to help aid future car buys when buying a car.

# To-Do List
 - Create README on github that allows for project understanding
 - Find a dataset for used cars and there pricing 
 - create a function to put it into a pandas a dataframe
 - wrangle and explore, create a model that allows us to predict car price.
 - deploy this model to a website for users to use
 
 
# Deliverables 
 - Final Notebook
 - Python scripts files 
 - Trello Board
 - Detailed 
 
# Executive Summary
### This project is an attempt to gain insight on the price prediction for used cars. This repository is an attempt to gain insight on the price prediction for used cars. This model will help shape a price prediction using a regression machine learning model to help aid future car buys when buying a car.
#### target = price 

# Hypothesis 
 - Year , make and model are key indicators for predicting price.
 - Color does not play a major for role in price.
 
 
# Key Takeaways
 - Horsepower plays more of a role in price prediction then originally thought
 - Training RMSE Score: 5345.38
 - Validate RMSE Score: 5046.09
 - Test     RMSE Score: 4974.11
 
# Data Dictionary


|  Variables             |  Definition                                |  Data Type             |
| :--------------------: | :----------------------------------------: | :--------------------: |
|  vin                   |  unique vin number                         |  object                |
|  back_legroom          |  legroom in the back seats (in)            |  float                 |
|  body_type             |  number of bathrooms for the property      |  object                |
|  city                  |  car city location                         |  object                |
|  city_fuel_economy     |  mpg for city driving                      |  float                 |
|  daysonmarket          |  number of days on the market              |  integer               |
|  dealer_zip            |  dealer zip code                           |  integer               |
|  engine_displacement   |  Engine displacment                        |  integer               |
|  engine_type           |  type of engine                            |  object                |
|  exterior_color        |  color of the exterior                     |  object                |
|  fleet                 |  whether the cars been a fleet car         |  bool                  |
|  frame_damaged         |  whether the car has frame damage          |  bool                  |
|  franchise_dealer      |  whether the seller us a franchise dealer  |  bool                  |
|  front_legroom         |  legroom in front (inches)                 |  float                 |
|  fuel_tank_volume      |  tank size.                                |  float                 |
|  fuel_type             |  fuel type                                 |  object                |
|  has_accidents         |  Whether that car has an accident on file  |  bool.                 |
|  height                |  height of the car                         |  float.                |
|  highway_fuel_economy  |  highway mpg                               |  float                 |
|  horsepower            |  length of the car                         |  float                 |
|  interior_color        |  interior color                            |  object                |
|  is_new                |  whether the car isa new listin            |  bool                  |
|  latitude              |  latitude of the car                       |  object                |
|  length                |  length of the car                         |  float                 |
|  listing_color         |  color listed on cargurus                  |  object                |
|  longitude             |  longitude if dealership location          |  object                |
|  make_name             |  name of vehicle                           |  object                |
|  maximum_seating       |  number of seats in vehicle                |  integer               |
|  mileage               |  miles on odometer                         |  integer               |
|  model_name            |  the date the house was sold               |  datetime              |
|  owner_count           |  Previous owners                           |  integer               |
|  wheel_system          |  Drive train type                          |  object                |
|  width                 |  width of the car                          |  float                 |
|  year                  |  car year                                  |  integer               |

 
# Recreate this project
- Download this repository
- Run Final Notebook
- pip install streamlit
- enter streamlit run app.py to make the website local
 
 



