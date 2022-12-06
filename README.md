# StanODEIntro
Three stan models that fit two parameters to a logistic model, all implemented in different ways. 

These examples were executed using CMDStanR -- additional packages are included in the R file, but will need to be downloaded from CRAN before use in R by using the "include.packages("...")" command. 

These examples are here only to accompany a blog post -- if/when this is accepted I will link to the post here which will give everything some more context. 

The files included are a csv file with some dummy data generated using the ipynb file included, an R file which you can use straight out of the box to run everything (once file locations are added at the appropriate points and any relevant packages are installed), and three annotated Stan files that will take the data from the provided csv file and fit a logistic curve to it, each varying in implementation. 
