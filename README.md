# Volpes Energy EV Energy Management System
This backend services optimises the charging schedule for a fleet of electric vehicles. The operator of the fleet submits high-level constraints on charging requirements such as expected departure time and required state of charge at departure. The backend collects external data such as power prices and demand forecasts for the residual load at the site. 

The inputs are combined into a linear optimisation program and solved by glpk to find the lowest-cost charging schedule for the fleet of vehicles.

The formulation is agnostic to the type of vehicle, but parameters such as charging speed and battery capacity can be individually provided.

## Technology
The backend is written in Python, using Pyomo for the model formulation and glpk as solver. Versioning and CI/CD is realised through github. The backend runs on Google Cloud, using serverless.

## Inputs and outputs
For this MVP, following inputs and outputs are used [_plans for future integration_]

#### Inputs
* Charging requirements: Input through web interface, send as post request
* Day-ahead power prices: Read from a local node returning .json [_connection to NordPool restful API, currently omitted due to high subscription fees_] 
* Other parameters for charging location: Currently hard-coded. [_additional input form from web interface_]

#### Outputs
* Optimal charging schedule per bus for 24 hours, returned as .json
* Cost savings compared to "dumb-charging with peak limiter", returned as .json