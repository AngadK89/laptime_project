<h2>Project Overview</h2>
The goal of this project is to be able to predict the lap time of a given Formula 1 car around any track. However, rather than just limiting this to the pool of known Formula 1 race tracks, this will be abstracted out to any race track in general. To do so, we will use the following track information:
- Track length
- Number of corners
- Track rotation
- Number of DRS zones
- Spatial coordinates of each corner on track
Particularly, this last feature set is meant to provide the model with a rough geospatial map of the track, helping it understand the track layout.

Moreover, we will also supply further information to the model, such as weather information, namely:
- Track temperature
- Air temperature
- Rainfall 
- Air humidity 

And information about the lap we are trying to predict:
- Session type (Sprint/Race/Qualifying)
- Tyre compound 
- Tyre age 
- Track status


<h2>Project Process</h2>
In order to successfully complete this task, I will need to perform the following tasks:
<list>
 1. Extract necessary data using the FastF1 API (contains Formula 1 data), utilising the OpenAI API or other Formula 1 related APIs to fill in information that is not available in FastF1.
 2. Preprocess the data to perform feature selection, feature extraction, scaling, and transformations. This helps prepare the dataset for the model. 
 3. Experiment with different regression models and identify whichever works best, leading to minimal overfitting and maximising the accuracy of the model. 
 4. Optimising results using hyperparameter tuning and further data preprocessing modifications. 
</list>

<h2>Project Status</h2>
This is an ongoing project, and currently, I am in the midst of performing steps 2 & 3. I have implemented a basic Decision Tree Regressor as a baseline model. However, it is severely overfitting. Thus, I am now going back to the feature set and identifying how I can eliminate/modify features, particularly transforming the spatial coordinates of each corner, to prevent this.
