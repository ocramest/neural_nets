# library(devtools)
# install_github('rstudio/reticulate',force=T)
# library(reticulate)
# library(tensorflow)
# install_tensorflow(version= "1.1.0")
# install_github("rstudio/keras",force=T)
#   library(keras)
# keras::install_keras()


suppressMessages(library(tidyverse))
bd = iris
bd$Species = ifelse(bd$Species=="versicolor",1,0)
muestra = sample(1:nrow(bd), size = round(.8*nrow(bd)), replace = F)
train = bd[muestra,]
test = bd[-muestra,]
response_train = data.frame(success = rep(NA, nrow(train)), failure = rep(NA, nrow(train)))
response_test = data.frame(success = rep(NA, nrow(test)), failure = rep(NA, nrow(test)))
response_train = response_train %>% 
  mutate(success = ifelse(train$Species==1, 1, 0),
         failure = ifelse(train$Species==1, 0, 1))
response_test = response_test %>% 
  mutate(success = ifelse(test$Species==1, 1, 0),
         failure = ifelse(test$Species==1, 0, 1))
predictors_train = bd[muestra,-5]
predictors_test = bd[-muestra,-5]

library(keras)
use_condaenv("r-tensorflow")
modelo = keras_model_sequential()

modelo %>% 
  layer_dense(name = "DeepLayer1",
              units = 4,
              activation = "relu",
              input_shape = c(4)) %>% 
  layer_dense(name = "DeepLayer2",
              units = 4,
              activation = "relu") %>% 
  layer_dense(name = "OutputLayer",
              units = 2,
              activation = "softmax")

summary(modelo)

modelo %>% compile(loss = "categorical_crossentropy",
                  optimizer = "adam",
                  metrics = c("accuracy"))


predictors_train = as.matrix(predictors_train)
response_train = as.matrix(response_train)
predictors_test = as.matrix(predictors_test)
response_test = as.matrix(response_test)

library(reticulate)
library(tensorflow)
history <- modelo %>% 
  fit(predictors_train,
      response_train,
      epoch = 10,
      batch_size = 256,
      validation_split = 0.2,
      verbose = 2)

modelo %>% 
  evaluate(predictors_test,
           response_test)

pred = modelo %>% 
  predict(predictors_test) %>% 
  `>`(0.5)

table(predicho = pred,
      real = response_test)


###### otro ejemplo:
cosa = rnorm(1e4)
cosa2 = rexp(1e4)
response = ifelse(cosa>-.4 & cosa2<1, round(runif(1e4)), 0)

datos = data.frame(cosa, cosa2, response)

muestra = sample(1:1e4, size = 9500, replace = FALSE)

train_x = datos[muestra, -3] %>% as.matrix()
train_y = datos[muestra, 3] %>% as.matrix()
train_y = cbind(ifelse(train_y==1, 1, 0), ifelse(train_y==1, 0, 1))

test_x = datos[-muestra,-3] %>% as.matrix()
test_y = datos[-muestra, 3] %>% as.matrix()
test_y = cbind(ifelse(test_y==1, 1, 0), ifelse(test_y==1, 0, 1))



modelo = keras_model_sequential()

modelo %>% 
  layer_dense(name = "DeepLayer1",
              units = 2,
              activation = "relu",
              input_shape = c(2)) %>% 
  layer_dense(name = "DeepLayer2",
              units = 2,
              activation = "relu") %>% 
  layer_dense(name = "OutputLayer",
              units = 2,
              activation = "softmax")

modelo %>% compile(loss = "categorical_crossentropy",
                   optimizer = "adam",
                   metrics = c("accuracy"))

hist <- modelo %>% 
  fit(train_x,
      train_y,
      epoch = 20,
      batch_size = 512,
      validation_split = 0.2,
      verbose = 2)

modelo %>% 
  evaluate(test_x,
           test_y)

pred = modelo %>% 
  predict(test_y) %>% 
  `>`(0.5)

table(predicho = pred,
      real = test_y)
