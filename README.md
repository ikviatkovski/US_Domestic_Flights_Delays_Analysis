
# US Domestic Flights Delays Analysis #
***
### *Capstone Project for Data Science Immersive Course at GENERAL ASSEMBLY, NYC* ###
***
## Problem Statement ##

Nowadays, flight delays have become a part of our daily lives. As much as we take them as an integral part of traveling, we would naturally like to avoid them. But eliminating flight delays entirely is not possible due to quite a large stochastic element of modern airline business - transportation volumes increase every year with more and more planes  taking off and landing every single minute, airport infrastructure is slow to adjust and costly to improve, weather and human factors impact on-time flight performance at random, security issues are on the rise, just to name a few.

But even if we cannot eliminate flight delays entirely, we could possibly try minimizing their negative impact. No one raises a brow when a flight is just "fashionably” late, making it to the gate still on time for even the tightest connections, but what if that delay becomes lengthier? It's an industry standard now in the US to consider the flight arriving at its disembarkation position no later than 15 minutes after it's scheduled time as still being 'on-time'. 

According to Federal Aviation Authority's estimate, the annual costs of delays (direct cost to airlines and passengers, lost demand, and indirect costs)  in 2018 was around USD28 billion. Indirect effect of flight delays is imposed customer costs, as well as lost productivity, wages and goodwill. According to one of the industry's watchdogs Airline for America, *”In 2018, the average cost of aircraft block (taxi plus airborne) time for U.S. passenger airlines was USD74.20 per minute, 9 percent more than in 2017. Fuel costs, the largest line item, rose 27 percent to USD27.01 per minute. Crew costs, the second largest line item, rose 3 percent to USD23.35 per minute.”*

**Could we possibly try to analyze historical flight data to understand the reasons for lengthier delays and try to predict such occurrences in order to take proactive measures minimizing their negative impact?**
***

## Project Files ##

This Project consists of a number of separate Jupyter Notebooks listed in the order of their workflow sequence and available at the __[code](http://localhost:8888/tree/PROJECTS/cstone/code)__ folder in the Project repository:

* __[1 Importing Data](http://localhost:8888/notebooks/PROJECTS/cstone/code/1%20Importing%20data.ipynb)__ - a notebook with initial data import from internet databases
* __[2 Cleaning](http://localhost:8888/notebooks/PROJECTS/cstone/code/2%20Cleaning.ipynb)__ - a notebook with the code for data cleaning and some insight on the initial data structure
* __[3 EDA](http://localhost:8888/notebooks/PROJECTS/cstone/code/3%20EDA.ipynb)__ - a notebook with the extensive Exploratory Data Analysis
* __[4 Pre-Processing](http://localhost:8888/notebooks/PROJECTS/cstone/code/4%20Pre-Processing.ipynb#Baseline-Model)__ - a notebook with data preparation for modeling and setting the Baseline model for classification
* __[5 Modeling Logistic Regression](http://localhost:8888/notebooks/PROJECTS/cstone/code/5%20Modeling%20Logistic%20Regression.ipynb#Importing-Data-and-Initial-Checks)__ - a notebook with Logistic Regression Classification model
* __[6 Modeling Decision Trees](http://localhost:8888/notebooks/PROJECTS/cstone/code/6%20Modeling%20Decision%20Trees.ipynb)__ - a notebook with Decision Trees family models
* __[7 Modeling FFNN Classifier](http://localhost:8888/notebooks/PROJECTS/cstone/code/7%20Modeling%20FFNN%20%20Classifier.ipynb)__ - a notebook with Forward Feeding Neural Network Classification model
* __[8 Modeling FFNN Regressor](http://localhost:8888/notebooks/PROJECTS/cstone/code/8%20Modeling%20FFNN%20Regressor.ipynb)__ - a notebook with Forward Feeding Neural Network Regression model

Datasets and auxiliary data files used for the Project and generated through the Project exceed any available external storage and are stored locally only.

[AUX](http://localhost:8888/tree/PROJECTS/cstone/AUX) folder contains a Jupyter notebook [Auxiliary Code](http://localhost:8888/notebooks/PROJECTS/cstone/AUX/Auxiliary%20Code.ipynb) used for visuals creation on features importance visualizations, and a mix of auxiliary DataFrames created through the process, as well as png files of all visualizations for the Project presentation. 
***
## Data Dictionary ##

Initial datasets used for the Project were obtained in the [Data Library of Bureau of Tranportation Statistics](https://www.transtats.bts.gov/Databases.asp?Mode_ID=1&Mode_Desc=Aviation&Subject_ID2=0) as a part of US Departament of Transportation and US Federal Aviation Authority Open Data Project. 
Flight performance information is available for download per monthly reporting periods with possibility to pre-select parameters of interest.
A data dictionary for all parameters is available at [Data Dictionary](https://www.transtats.bts.gov/Tables.asp?DB_ID=120&DB_Name=Airline%20On-Time%20Performance%20Data&DB_Short_Name=On-Time) where all the fields and industry-specific terminology is explained in the sections corresponding to **"Reporting Carrier On-Time Performance"** section.
Through the Project flow some additional explanations are given wherever necessary.
***
## Executive Summary ##

In order to analyze flight delays pattern I decided to pick the most recent period of Quarter 4, 2018. The last three months of each year are quite representative in terms of domestic travel - October is quite diverse weather-wise and includes school vacations nearly everywhere, November is even more diverse weather-wise with increased winter storm disruptions and it also includes one of the highest travel period - Thanksgiving holidays. December is traditionally the most complicated month as weather always brings unpleasant surprises, it includes Christmas as another school vacations time everywhere in the US and another busiest time across the airports. 

Initial monthly domestic travel data file contain around 600_000 flights each month. The resulting databases are humongous and for the sake of simplification, speeding up of work flow and (!) ease of debugging the Project was divided into several separate Jupiter notebooks with unavoidably repetitive beginnings. 

Workflow explanations are given in accordance with the Project notebooks organization order. 

### 1. Importing Data ###

Data was initially spread between the three monthly flight statistic files for October, November and December 2018 accordingly. In this Jupyter Notebook I did basic exploration of these csv files and corresponding Pandas DataFrames and eventually concatenated them into a joint csv file. 

### 2. Cleaning ###

Data acquired from an external source naturally weren't ideally suited for planned research. In this notebook I did some standard data cleaning, as well as significantly reduced the amount of data features as most of them either didn't relate to the research or repeated other data set features. Also I had to change some data formats and address missing values resulted from data structure, as well as to take conceptual decisions to drop cancelled flights from the research entirely (they represent a different subset of flights and a topic for another possible research) and keep diverted flights in the datasets only if they eventually reached their final destination (not being cancelled at the diversion airport, as the ones that didn't reach their final destination and were cancelled at the diversion airport represent another distinct very small subset and dropping them in favor of simplifying research methodology made sense).
Also, for the reasons of being more informative to the research' audience, I had to replace air-carrier codes with their respected full names.
As well, I had to tackle some outliers - both positive and negative - so that the remaining dataset trends weren't influenced by them. 

### 3. EDA ###

Explorative Data Analysis quite straight-forward.
For the reasons of research target audience's convenience I had to re-attribute smaller feeder airlines flight to the larger carriers who actually act as a ticket seller and main point of customer's contact. Then I explored average delays by carrier and their reasons, as well as delays by reason breakdown.

*It is important to note that the flights reported to FAA have several categories of delay, as well as total delay data. A sum of these delay categories is not always equal to the flight's total delay, and represents the **explained flight delay** as some other parts of the total delay might be either not reported into any of these categories, or there's no categories breakdown at all. This is a potential source of confusion.*

Finally, in this section I had a look at our target variable distribution. 

### 4. Pre-Processing ###

The purpose of this separate notebook is to create a dataset to be universally used for any modeling in future.

In order to do so I looked at correlations between our remaining features, as well as dealt with imbalanced classes problem for classification models. To avoid imbalanced classes with our future classification models I chose to construct my modeling dataset in a way that the positive and negative classes are balanced as 1:1. I randomly pulled observations out of our positive (flight on time) class - and a number of these observations will be equal to the number of tweets in our negative (flight delayed) class.

### 5 Logistic Regression Classifier ###

Fitting a Logistic Regression Classifier on flights data is a classic approach to classifications problems. 

In my case it resulted in default model performing with accuracy of slightly above 0.59 which is just a light improvement for the Baseline model for balanced classed accuracy of 0.5.

I attempted hyper-parameters tuning and achieved just a very small, nearly negligible accuracy improvement.

### 6 Decision Trees ###

A default Decision Tree Classifier performed with accuracy of 1.0 on a training set, but just with 0.62 on a testing set. A Bagging Classifier showed similar results with training set accuracy of 0.98 and just 0.66 on a testing sets. Both models were showing signs of being severely overfit.

A default Random Forest model showed signs of being very overfit performing with the accuracy of 0.99 on a train-set and 0.64 on a test-set. Hyper-parameters grid-search allowed me to get rid of overfitting at a price of having my best grid-searched model accuracy of 0.60 on a training set and 0.59 on a testing set. This is not a significant improvement with both tuned Logistic Regression and Baseline models.

Fitting an AdaBoost Classifier with my best grid-searched Random Forest model as a base estimator permitted me to achieve accuracy of 0.70 on a training set and 0.65 on a testing set.The model is slightly overfit. 

Applying XGBoost technique resulted in accuracy of 0.65 on a training set and 0.64 on a testing set. Using GridSearch for its hyper-parameters tuning I achieved nearly the same results at cost of nearly a whole night of computations.

Hence, AdaBoost Classifier using as a base estimator an optimized Random Forest model showed the highest accuracy scores both on training and testing data. Due to its high computational complexity I could not yet have time to attempt its parameters tuning via GridSearch.

Aslo, I was able to use my best performing model to determine the most important flight delay predictors. They turned out to be flight data (month, date and weekday, flight distance, flight number and operating carrier, as well as departure/arrival from certain airports. More information with some visuals is available in the Project presentation. 

An interesting fact is that the most powerful flight delay predictor among flight destinations is the airport of Newark, NJ and Chicago O'Hare airport among flight origins, respectably.

### 7 Forward Feeding Neural Network Classifier ###

Fitting a FFNN with two hidden layers of 128 and 64 neurons resulted in max classification accuracy (after an early stop at 17 epoch) of 0.68 and validation accuracy of 0.64. This is slightly outperformed by the AdaBoost Classifier.

### 8 Forward Feeding Neural Network Regressor ###

As an attempts to forecast a delayed flight's delay duration I ran a FFNN Regressor model with 2 hidden layers of 512 and 256 neurons respectably, employing an Early Stop regularization technique. This resulted in the model stopping at the epoch 7 with the best achieved Mean Squared Error of roughly 640. With a baseline prediction of a mean delay of around 40 minutes, this corresponds to Root Mean Squared Error of 25, meaning that we could predict a delayed flight's actual delay length within +/- 25 minutes interval. Looking back at an average delay of 40 minutes - for that particular case we will be able to predict an interval of 15 to 65 minutes - which is quite far from satisfactory and leaves a lot of room for future improvements. 

## Conclusions ##

My attempt to build a predictive model for US domestic flights delays basing on historical performance information has resulted in highest achieved accuracy of 0.64 (or 64% ) on unseen data. That can be explained as my model correctly predicting roughly 2 out of 3 flight delays correctly.
A regression model trying to predict a duration of delay for a delayed flight didn't perform at a satisfactory level, as its variance was too high.
Analyzing the factors influencing the possible delay the highest I was able to identify that the strongest predictors were the day of the week, the flight departure hour and the flight distance. I was also able to identify the list of airports among the flight origins and destinations influencing possible delay the most (and, to no surprise, most of them were on both lists). 


## Future Steps ##

This project has had restraints both in terms of time and available computational resource. It has also helped identify a number of possible improvement and further research, namely:

* Classification models need to be further tuned
* Building and further tuning of regression models based on Decision Trees family, Support Vectors, etc. in order to get viable flight delay duration predictions
* There's plenty of room to deep-diving into feature engineering for the delay information contained in the dataset's fields. As possible directions I would suggest a closer look at the explained delays breakdowns in order to identify possible links to strong predictive features (one of them being linking aircraft tail number with the flights it's making per day, possible extraction of aircraft type influence on the delay, etc. )
* Weather factor input into modeling. For that purpose using historical weather databases API's would be quite beneficial and will most probably lead to performance improvements.
* Additional research and incorporation into the datasets information on the international flights in the US airspace (landing in the US and intersecting it without a landing), but this would dramatically broaden the Project's scope and will definitely require a serious technical base upgrade (definitely, at least GPU use).

## Source Documentation ##

Images used in Project Presentation are either created within the Project or sourced at **[Wikimedia Commons](https://commons.wikimedia.org/wiki/Main_Page/)** and are copyright-free.

Whenever appropriate, code credits are given in code comments.




